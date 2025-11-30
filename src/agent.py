"""
Agentic Assistant - SQL + RAG System
Intelligently routes between SQL queries and RAG retrieval for automotive data
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import traceback
from datetime import datetime
import duckdb 
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DATABASE_PATH,
    CSV_DIR,
    RELOAD_DATA,
    print_config_status,
)
from prompts import (
    SQL_SCHEMA,
    format_sql_generation_prompt,
    format_sql_results_prompt,
    format_routing_prompt,
    format_hybrid_combine_prompt,
)
from rag import RAGTool
from utils import call_llm
from guardrails import (
    validate_user_input,
    validate_sql_query,
    validate_llm_output,
    get_safe_response,
    GuardrailViolation,
)

print_config_status()


class AgentExecutionError(Exception):
    """Custom exception that carries the partial trace"""
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


class GuardrailError(Exception):
    """Exception raised when guardrails block a request"""
    def __init__(self, message: str, violation_type: GuardrailViolation):
        super().__init__(message)
        self.violation_type = violation_type


@dataclass
class QueryResult:
    """Container for query results"""
    answer: str
    tools_used: List[str]
    sql_query: Optional[str] = None
    citations: List[str] = None
    execution_time_ms: float = 0
    trace: List[Dict] = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    blocked_by_guardrail: bool = False


class SQLTool:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.schema = SQL_SCHEMA
    
    def generate_sql(self, question: str, client: OpenAI) -> tuple[str, int, float]:
        """Use GPT to generate SQL from natural language
        
        Returns:
            Tuple of (sql_query, tokens_used, cost_usd)
            
        Raises:
            GuardrailError: If generated SQL fails safety checks
        """
        prompt = format_sql_generation_prompt(question)
        response = call_llm(client, prompt, verbosity_level="low", reasoning_level=None, return_usage=True)
        
        # Clean up LLM response - remove markdown and any preamble text
        sql = response.text.replace("```sql", "").replace("```", "").strip()
        
        # Remove any text before SELECT/WITH (LLM sometimes adds explanations)
        select_match = re.search(r'\b(SELECT|WITH)\b', sql, re.IGNORECASE)
        if select_match:
            sql = sql[select_match.start():]
        
        # Validate generated SQL
        sql_check = validate_sql_query(sql)
        if not sql_check.passed:
            print(f"\n{'='*60}")
            print(f"SQL GUARDRAIL BLOCKED")
            print(f"{'='*60}")
            print(f"GENERATED SQL:\n{sql}")
            print(f"{'='*60}")
            print(f"VIOLATION: {sql_check.violation_type.value}")
            print(f"MESSAGE: {sql_check.message}")
            print(f"DETAILS: {sql_check.details}")
            print(f"{'='*60}\n")
            raise GuardrailError(sql_check.message, sql_check.violation_type)
        
        return sql, response.total_tokens, response.calculate_cost()
    
    def execute(self, sql: str) -> List[Dict]:
        """Execute SQL using DuckDB"""
        conn = duckdb.connect(self.db_path)
        
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in rows]
            else:
                results = []
                
            return results
            
        except Exception as e:
            raise Exception(f"DuckDB execution failed: {str(e)}")
        finally:
            conn.close()
    
    def format_results(self, results: List[Dict], question: str, client: OpenAI) -> tuple[str, int, float]:
        """Format SQL results into natural language answer
        
        Returns:
            Tuple of (answer, tokens_used, cost_usd)
        """
        prompt = format_sql_results_prompt(question, json.dumps(results, indent=2, default=str))
        response = call_llm(client, prompt, verbosity_level="medium", reasoning_level=None, return_usage=True)
        return response.text, response.total_tokens, response.calculate_cost()


class AgentOrchestrator:
    """Routes queries to appropriate tools"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.sql_tool = SQLTool()
        self.rag_tool = RAGTool()
    
    def route_query(self, question: str) -> tuple[str, int, float]:
        """Determine which tool(s) to use
        
        Returns:
            Tuple of (routing_decision, tokens_used, cost_usd)
        """
        prompt = format_routing_prompt(question)
        response = call_llm(self.client, prompt, verbosity_level="low", reasoning_level=None, return_usage=True)
        return response.text.upper(), response.total_tokens, response.calculate_cost()
    
    def _elapsed_ms(self, start: datetime) -> float:
        """Calculate elapsed time in milliseconds"""
        return round((datetime.now() - start).total_seconds() * 1000, 1)
    
    def process(self, question: str) -> QueryResult:
        """Main entry point - routes and executes query"""
        start_time = datetime.now()
        trace = []
        total_tokens = 0
        total_cost = 0.0
        
        # Input guardrail check
        input_check = validate_user_input(question)
        if not input_check.passed:
            trace.append({
                "tool": "Input Guardrail",
                "elapsed_ms": self._elapsed_ms(start_time),
                "input": {"question": question},
                "output": {
                    "blocked": True,
                    "violation_type": input_check.violation_type.value,
                    "reason": input_check.message
                }
            })
            return QueryResult(
                answer=get_safe_response(input_check.violation_type),
                tools_used=["Input Guardrail"],
                trace=trace,
                execution_time_ms=self._elapsed_ms(start_time),
                blocked_by_guardrail=True
            )
        
        try:
            step_start = datetime.now()
            routing_decision, tokens, cost = self.route_query(question)
            total_tokens += tokens
            total_cost += cost
            trace.append({
                "tool": "Query Router",
                "elapsed_ms": self._elapsed_ms(step_start),
                "tokens": tokens,
                "cost_usd": round(cost, 6),
                "input": {"question": question},
                "output": {"query_type": routing_decision}
            })
            
            if routing_decision == "SQL":
                result, step_tokens, step_cost = self._handle_sql(question, trace)
            elif routing_decision == "RAG":
                result, step_tokens, step_cost = self._handle_rag(question, trace)
            else:
                result, step_tokens, step_cost = self._handle_hybrid(question, trace)
            
            total_tokens += step_tokens
            total_cost += step_cost
            
            # Output guardrail check
            output_check = validate_llm_output(result.answer, question)
            if not output_check.passed:
                trace.append({
                    "tool": "Output Guardrail",
                    "elapsed_ms": self._elapsed_ms(start_time),
                    "input": {"answer_length": len(result.answer)},
                    "output": {
                        "blocked": True,
                        "violation_type": output_check.violation_type.value,
                        "reason": output_check.message
                    }
                })
                return QueryResult(
                    answer=get_safe_response(output_check.violation_type),
                    tools_used=result.tools_used + ["Output Guardrail"],
                    sql_query=result.sql_query,
                    trace=trace,
                    execution_time_ms=self._elapsed_ms(start_time),
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    blocked_by_guardrail=True
                )
            
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = elapsed
            result.trace = trace
            result.total_tokens = total_tokens
            result.total_cost_usd = total_cost
            return result
        
        except GuardrailError as e:
            trace.append({
                "tool": "SQL Guardrail",
                "elapsed_ms": self._elapsed_ms(start_time),
                "input": {"question": question},
                "output": {
                    "blocked": True,
                    "violation_type": e.violation_type.value,
                    "reason": str(e)
                }
            })
            return QueryResult(
                answer=get_safe_response(e.violation_type),
                tools_used=["SQL Guardrail"],
                trace=trace,
                execution_time_ms=self._elapsed_ms(start_time),
                blocked_by_guardrail=True
            )

        except Exception as e:
            trace.append({
                "tool": "Error",
                "elapsed_ms": self._elapsed_ms(start_time),
                "input": {"question": question},
                "output": {"error": str(e), "details": traceback.format_exc()}
            })
            raise AgentExecutionError(f"Step failed: {traceback.format_exc()}", trace) from e
    
    def _handle_sql(self, question: str, trace: List) -> tuple[QueryResult, int, float]:
        """Handle SQL-only queries
        
        Returns:
            Tuple of (QueryResult, total_tokens, total_cost)
        """
        total_tokens = 0
        total_cost = 0.0
        
        step_start = datetime.now()
        sql, tokens, cost = self.sql_tool.generate_sql(question, self.client)
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "SQL Generator (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"question": question},
            "output": {"sql_query": sql}
        })
        
        try:
            step_start = datetime.now()
            results = self.sql_tool.execute(sql)
            trace.append({
                "tool": "SQL Executor (DuckDB)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"sql_query": sql},
                "output": {"rows_returned": len(results), "data": results[:5] if len(results) > 5 else results}
            })
        except Exception as e:
            trace.append({
                "tool": "SQL Executor (DuckDB)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"sql_query": sql},
                "output": {"error": traceback.format_exc(), "status": "failed"}
            })
            raise e
        
        step_start = datetime.now()
        answer, tokens, cost = self.sql_tool.format_results(results, question, self.client)
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "Result Formatter (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"question": question, "rows_count": len(results)},
            "output": {"answer": answer}
        })
        
        return QueryResult(
            answer=answer,
            tools_used=["SQL Query Tool"],
            sql_query=sql,
            citations=["FACT_SALES", "DIM_MODEL", "DIM_COUNTRY"]
        ), total_tokens, total_cost
    
    def _handle_rag(self, question: str, trace: List) -> tuple[QueryResult, int, float]:
        """Handle RAG-only queries
        
        Returns:
            Tuple of (QueryResult, total_tokens, total_cost)
        """
        total_tokens = 0
        total_cost = 0.0
        
        step_start = datetime.now()
        docs = self.rag_tool.retrieve(question)
        trace.append({
            "tool": "Document Retriever (Embedding)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "input": {"question": question},
            "output": {
                "docs_found": len(docs),
                "sources": [{"source": d.source, "page": d.page, "score": round(score, 4)} for d, score in docs]
            }
        })
        
        step_start = datetime.now()
        answer, tokens, cost = self.rag_tool.answer(question, docs, self.client)
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "RAG Answer Generator (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"question": question, "context_docs": len(docs)},
            "output": {"answer": answer}
        })
        
        return QueryResult(
            answer=answer,
            tools_used=["RAG - Document Retrieval"],
            citations=[f"{d.source} (page {d.page}), relevance score {score:.4f}" for d, score in docs]
        ), total_tokens, total_cost
    
    def _handle_hybrid(self, question: str, trace: List) -> tuple[QueryResult, int, float]:
        """Handle queries needing both SQL and RAG
        
        Returns:
            Tuple of (QueryResult, total_tokens, total_cost)
        """
        total_tokens = 0
        total_cost = 0.0
        
        step_start = datetime.now()
        sql, tokens, cost = self.sql_tool.generate_sql(question, self.client)
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "SQL Generator (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"question": question},
            "output": {"sql_query": sql}
        })
        
        try:
            step_start = datetime.now()
            sql_results = self.sql_tool.execute(sql)
            trace.append({
                "tool": "SQL Executor (DuckDB)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"sql_query": sql},
                "output": {"rows_returned": len(sql_results), "data": sql_results[:5] if len(sql_results) > 5 else sql_results}
            })
        except Exception as e:
            trace.append({
                "tool": "SQL Executor (DuckDB)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"sql_query": sql},
                "output": {"error": traceback.format_exc(), "status": "failed"}
            })
            raise e
        
        step_start = datetime.now()
        sql_answer, tokens, cost = self.sql_tool.format_results(sql_results, question, self.client)
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "SQL Result Formatter (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"question": question, "rows_count": len(sql_results)},
            "output": {"intermediate_answer": sql_answer}
        })
        
        try:
            step_start = datetime.now()
            docs = self.rag_tool.retrieve(question)
            trace.append({
                "tool": "Document Retriever (Embedding)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"question": question},
                "output": {
                    "docs_found": len(docs),
                    "sources": [{"source": d.source, "page": d.page, "score": round(score, 4)} for d, score in docs]
                }
            })
            
            step_start = datetime.now()
            rag_answer, tokens, cost = self.rag_tool.answer(question, docs, self.client)
            total_tokens += tokens
            total_cost += cost
            trace.append({
                "tool": "RAG Answer Generator (LLM)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "tokens": tokens,
                "cost_usd": round(cost, 6),
                "input": {"question": question, "context_docs": len(docs)},
                "output": {"intermediate_answer": rag_answer}
            })
            
        except Exception as e:
            trace.append({
                "tool": "Document Retriever (Embedding)",
                "elapsed_ms": self._elapsed_ms(step_start),
                "input": {"question": question},
                "output": {"error": traceback.format_exc(), "status": "failed"}
            })
            raise e

        step_start = datetime.now()
        combined_prompt = format_hybrid_combine_prompt(question, sql_answer, rag_answer)
        response = call_llm(self.client, combined_prompt, verbosity_level="medium", reasoning_level=None, return_usage=True)
        answer = response.text
        tokens = response.total_tokens
        cost = response.calculate_cost()
        total_tokens += tokens
        total_cost += cost
        trace.append({
            "tool": "Hybrid Combiner (LLM)",
            "elapsed_ms": self._elapsed_ms(step_start),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "input": {"sql_answer_length": len(sql_answer), "rag_answer_length": len(rag_answer)},
            "output": {"final_answer": answer}
        })
        
        return QueryResult(
            answer=answer,
            tools_used=["SQL Query Tool", "RAG - Document Retrieval"],
            sql_query=sql,
            citations=[f"{d.source} (page {d.page}), relevance score {score:.4f}" for d, score in docs]
        ), total_tokens, total_cost


def setup_database(db_path: str = DATABASE_PATH, force_reload: bool = None):
    """
    Load CSV data directly into DuckDB tables.
    Uses provided CSV columns (Units only).
    
    Args:
        db_path: Path to the database file
        force_reload: If True, recreate database. If None, uses RELOAD_DATA config.
    """
    reload = force_reload if force_reload is not None else RELOAD_DATA
    
    if os.path.exists(db_path) and not reload:
        print(f"✓ Database already exists at {db_path} (set RELOAD_DATA=true to rebuild)")
        return
    
    if not CSV_DIR.exists():
        print(f"❌ Error: Data directory not found at {CSV_DIR}")
        return

    if reload and os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✓ Deleted existing database (RELOAD_DATA=true)")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete existing database: {e}")

    print(f"Initializing DuckDB from CSVs at {CSV_DIR}...")
    conn = duckdb.connect(db_path)
    
    try:
        conn.execute(f"CREATE TABLE DIM_MODEL AS SELECT * FROM read_csv_auto('{CSV_DIR / 'DIM_MODEL.csv'}')")
        
        conn.execute(f"""
            CREATE TABLE DIM_COUNTRY AS 
            SELECT country AS country_name, country_code, region 
            FROM read_csv_auto('{CSV_DIR / 'DIM_COUNTRY.csv'}')
        """)

        conn.execute(f"CREATE TABLE DIM_ORDERTYPE AS SELECT * FROM read_csv_auto('{CSV_DIR / 'DIM_ORDERTYPE.csv'}')")

        conn.execute(f"CREATE TABLE raw_sales AS SELECT * FROM read_csv_auto('{CSV_DIR / 'FACT_SALES.csv'}')")
        
        conn.execute("""
            CREATE TABLE FACT_SALES AS
            SELECT 
                model_id,
                country_code,
                make_date(year, month, 1) as sale_date,
                contracts as units
            FROM raw_sales
        """)
        
        conn.execute(f"CREATE TABLE raw_sales_ot AS SELECT * FROM read_csv_auto('{CSV_DIR / 'FACT_SALES_ORDERTYPE.csv'}')")

        conn.execute("""
            CREATE TABLE FACT_SALES_ORDERTYPE AS
            SELECT 
                model_id,
                country_code,
                ordertype_id,
                make_date(year, month, 1) as sale_date,
                contracts as units
            FROM raw_sales_ot
        """)

        conn.execute("DROP TABLE raw_sales")
        conn.execute("DROP TABLE raw_sales_ot")

        count = conn.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0]
        print(f"✓ Database loaded successfully.")
        print(f"✓ Loaded {count} sales records (Units only).")

    except Exception as e:
        print(f"❌ Error loading CSVs: {e}")
        raise e
    finally:
        conn.close()


def main():
    """Main CLI interface"""
    print("=" * 70)
    print("  AGENTIC ASSISTANT - SQL + RAG System")
    print("=" * 70)
    print()
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with: OPENAI_API_KEY=your-key-here")
        return
    
    print("✓ Loaded OpenAI API key from .env")
    print(f"✓ Using model: {OPENAI_MODEL}")
    
    setup_database()
    
    print(f"✓ Initializing agent with {OPENAI_MODEL}...")
    agent = AgentOrchestrator(OPENAI_API_KEY)
    
    examples = [
        "What were the monthly RAV4 HEV sales in Germany in 2024?",
        "What is the standard Toyota warranty for Europe?",
        "Where is the tire repair kit located in the Corolla?",
        "Compare Toyota vs Lexus SUV sales in Western Europe and summarize warranty differences"
    ]
    
    print("\nExample Questions:")
    for i, q in enumerate(examples, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        print("-" * 70)
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        if query.isdigit() and 1 <= int(query) <= len(examples):
            query = examples[int(query) - 1]
            print(f"Selected: {query}")
        
        print("\n🔍 Processing...")
        
        try:
            result = agent.process(query)
            
            print("\n" + "=" * 70)
            print("RESULT")
            print("=" * 70)
            print(f"\n📊 Tools Used: {', '.join(result.tools_used)}")
            print(f"⏱️  Execution Time: {result.execution_time_ms:.0f}ms")
            
            if result.sql_query:
                print(f"\n💾 SQL Query:\n{result.sql_query}")
            
            print(f"\n💬 Answer:\n{result.answer}")
            
            if result.citations:
                print(f"\n📚 Citations:")
                for citation in result.citations:
                    print(f"  - {citation}")
            
            print(f"\n🔬 Trace:")
            for step in result.trace:
                print(f"  {json.dumps(step, indent=4)}")
            
        except Exception:
            print(f"\n❌ Error: {traceback.format_exc()}")


if __name__ == "__main__":
    main()