"""
Streamlit UI for Agentic Assistant
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import traceback
from datetime import datetime

from config import OPENAI_API_KEY
from agent import AgentOrchestrator, setup_database, AgentExecutionError

# Page config
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'agent' not in st.session_state:
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY not found in .env file")
        st.info("Please create a .env file in the project root with: OPENAI_API_KEY=your-key-here")
        st.stop()
    
    with st.spinner("Setting up database..."):
        setup_database()
    st.session_state.agent = AgentOrchestrator(OPENAI_API_KEY)

# Header
st.title("🤖 Agentic Assistant")
st.markdown("**SQL + RAG System** for Automotive Data")
st.divider()

def display_trace(trace_data, trace_id: str = "main"):
    with st.expander("🔬 View Execution Trace (Tool Calls)", expanded=False):
        total_time = sum(step.get('elapsed_ms', 0) for step in trace_data)
        total_tokens = sum(step.get('tokens', 0) for step in trace_data)
        total_cost = sum(step.get('cost_usd', 0) for step in trace_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Total time: {total_time:.1f}ms**")
        with col2:
            st.markdown(f"**Total tokens: {total_tokens:,}**")
        with col3:
            st.markdown(f"**Total cost: ${total_cost:.4f}**")
        
        time_data = []
        for step in trace_data:
            if 'elapsed_ms' in step:
                time_data.append({"Tool": step.get('tool', 'Unknown'), "Time (ms)": step['elapsed_ms']})
        
        if time_data:
            import pandas as pd
            df = pd.DataFrame(time_data)
            st.bar_chart(df.set_index("Tool"))
        
        st.divider()
        
        for i, step in enumerate(trace_data, 1):
            tool_name = step.get('tool', 'Unknown Tool')
            elapsed = step.get('elapsed_ms', 0)
            
            st.markdown(f"### Step {i}: {tool_name}")
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Time", f"{elapsed:.1f}ms")
            with metric_cols[1]:
                tokens = step.get('tokens', 0)
                st.metric("Tokens", f"{tokens:,}" if tokens else "-")
            with metric_cols[2]:
                cost = step.get('cost_usd', 0)
                st.metric("Cost", f"${cost:.6f}" if cost else "-")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📥 Input:**")
                input_data = step.get('input', {})
                if 'question' in input_data:
                    st.text_area("Question", input_data['question'], height=68, disabled=True, key=f"q_{trace_id}_{i}")
                if 'sql_query' in input_data:
                    st.code(input_data['sql_query'], language='sql')
                for key, value in input_data.items():
                    if key not in ['question', 'sql_query']:
                        st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("**📤 Output:**")
                output_data = step.get('output', {})
                
                if 'query_type' in output_data:
                    st.success(f"Query Type: **{output_data['query_type']}**")
                if 'sql_query' in output_data:
                    st.code(output_data['sql_query'], language='sql')
                if 'error' in output_data:
                    st.error(f"Error: {output_data['error']}")
                if 'rows_returned' in output_data:
                    st.write(f"**Rows returned:** {output_data['rows_returned']}")
                if 'data' in output_data:
                    st.json(output_data['data'])
                if 'docs_found' in output_data:
                    st.write(f"**Documents found:** {output_data['docs_found']}")
                if 'sources' in output_data:
                    for src in output_data['sources']:
                        st.write(f"- {src['source']} (p.{src['page']}) - score: {src['score']}")
                if 'answer' in output_data:
                    st.info(output_data['answer'][:300] + "..." if len(output_data.get('answer', '')) > 300 else output_data['answer'])
                if 'intermediate_answer' in output_data:
                    st.info(output_data['intermediate_answer'][:200] + "..." if len(output_data.get('intermediate_answer', '')) > 200 else output_data['intermediate_answer'])
                if 'final_answer' in output_data:
                    st.success(output_data['final_answer'][:300] + "..." if len(output_data.get('final_answer', '')) > 300 else output_data['final_answer'])
            
            st.divider()

# Sidebar with examples
with st.sidebar:
    st.header("📋 Example Questions")
    
    examples = [
        "What were the monthly RAV4 HEV sales in Germany in 2024?",
        "What is the standard Toyota warranty for Europe?",
        "Where is the tire repair kit located in the Corolla?",
        "Compare Toyota vs Lexus SUV sales in Western Europe and summarize warranty differences",
        "What are the maintenance intervals for RAV4?",
        "Show me total Lexus sales by country"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"💡 {example[:50]}...", key=f"ex_{i}"):
            st.session_state.pending_query = example
    
    st.divider()
    
    st.header("ℹ️ System Info")
    st.markdown("""
    **Tools Available:**
    - 🗄️ SQL Query Tool
    - 📚 RAG Document Retrieval
    - 🔄 Hybrid Mode
    
    **Data Sources:**
    - Sales data (2024)
    - Contract documents
    - Owner's manuals
    """)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Check if there's a pending query from example buttons
if 'pending_query' in st.session_state:
    query_to_run = st.session_state.pending_query
    del st.session_state.pending_query
    
    # Display what's being searched
    st.text_input("Ask a question:", value=query_to_run, disabled=True)
    
    # Process immediately
    with st.spinner("🤔 Processing your question..."):
        try:
            result = st.session_state.agent.process(query_to_run)
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'query': query_to_run,
                'result': result,
                'status': 'success'
            })
            st.rerun()
        except AgentExecutionError as e:
            st.error(f"❌ Process Failed: {e}")
            st.warning("⚠️ Execution stopped due to an error. See trace below for details.")
            display_trace(e.trace, "pending_error")
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'query': query_to_run,
                'error': str(e),
                'trace': e.trace,
                'status': 'error'
            })
        except Exception as e:
            st.error(f"❌ Unexpected Error: {traceback.format_exc()}")

else:
    # Main interface - normal text input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question:",
            placeholder="e.g., What were the RAV4 sales in Germany last month?",
            key="query_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        submit = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if submit and query:
        with st.spinner("🤔 Processing your question..."):
            try:
                result = st.session_state.agent.process(query)
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'result': result,
                    'status': 'success'
                })
                st.rerun()
            except AgentExecutionError as e:
                st.error(f"❌ Process Failed: {e}")
                st.warning("⚠️ Execution stopped due to an error. See trace below for details.")
                display_trace(e.trace, "manual_error")
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'error': str(e),
                    'trace': e.trace,
                    'status': 'error'
                })
            except Exception as e:
                st.error(f"❌ Unexpected Error: {traceback.format_exc()}")

# Display results
if st.session_state.history:
    st.divider()
    
    # Show most recent result
    latest = st.session_state.history[-1]
    
    if latest['status'] == 'success':
        result = latest['result']
        
        # Check if blocked by guardrail
        if getattr(result, 'blocked_by_guardrail', False):
            st.subheader("🛡️ Request Blocked")
            st.warning("Your request was blocked by our safety guardrails.")
            
            st.markdown("### ❓ Query")
            st.text(latest['query'])
            
            st.markdown("### 💬 Response")
            st.info(result.answer)
            
            # Show trace for transparency
            display_trace(result.trace, "blocked")
        else:
            st.subheader("📊 Latest Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tools Used", len(result.tools_used))
            with col2:
                st.metric("Execution Time", f"{result.execution_time_ms:.0f}ms")
            with col3:
                cost_display = f"${result.total_cost_usd:.4f}" if result.total_cost_usd > 0 else "N/A"
                st.metric("Cost", cost_display, help=f"Total tokens: {result.total_tokens:,}")
            
            st.markdown("**🔧 Tools:** " + " → ".join(result.tools_used))
            
            st.markdown("### ❓ Query")
            st.text(latest['query'])
            
            if result.sql_query:
                with st.expander("💾 View SQL Query"):
                    st.code(result.sql_query, language="sql")
            
            st.markdown("### 💬 Answer")
            st.info(result.answer)
            
            if result.citations:
                with st.expander("📚 View Citations"):
                    for citation in result.citations:
                        st.markdown(f"- {citation}")
            
            # Use our helper function
            display_trace(result.trace, "latest")
        
    else:
        # History item was an error
        st.subheader("⚠️ Latest Attempt (Failed)")
        st.error(latest['error'])
        if 'trace' in latest:
            display_trace(latest['trace'], "error_history")
    
    # History
    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("📜 Query History")
        
        for i, item in enumerate(reversed(st.session_state.history[:-1]), 1):
            with st.expander(f"Query {len(st.session_state.history) - i}: {item['query'][:60]}..."):
                st.markdown(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                st.markdown(f"**Tools:** {', '.join(item['result'].tools_used)}")
                st.markdown(f"**Answer:** {item['result'].answer[:200]}...")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
</div>
""", unsafe_allow_html=True)