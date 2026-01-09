import streamlit as st
import json
import os
from typing import Dict
from datetime import datetime

# Import your existing classes
from agentic_context_engineer import AgenticCrew

# Page configuration
st.set_page_config(
    page_title="Agetic Context Engineer UI",
    page_icon="🤖",
    layout="wide"
)   

# Initialize session state
if 'ace_crew' not in st.session_state:
    # Auto-initialize from environment variable
    groq_api = os.getenv("GROQ_API", "")
    if groq_api:
        try:
            st.session_state.ace_crew = AgenticCrew(
                model="groq/qwen/qwen3-32b",
                api_key=groq_api,
                base_url=None
            )
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            st.stop()
    else:
        st.error("❌ GROQ_API environment variable not set!")
        st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #2ca02c;
    }
    .bullet-item {
        background-color: #f0f2f6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🤖 Agentic Context Engineer System</p>', unsafe_allow_html=True)
st.markdown("**Self-Improving Knowledge System with Dynamic Playbook**")

# Sidebar for status
with st.sidebar:
    st.header("📊 System Status")
    st.success("🟢 Active")
    
    playbook = st.session_state.ace_crew.playbook
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sections", len(playbook.sections))
    with col2:
        st.metric("Bullets", len(playbook.bullets))
    st.metric("Queries Processed", len(st.session_state.history))
    
    st.divider()
    
    # Quick stats
    if playbook.bullets:
        st.subheader("📈 Quick Stats")
        helpful = sum(b.helpful for b in playbook.bullets.values())
        harmful = sum(b.harmful for b in playbook.bullets.values())
        neutral = sum(b.neutral for b in playbook.bullets.values())
        
        st.write(f"✅ Helpful: {helpful}")
        st.write(f"❌ Harmful: {harmful}")
        st.write(f"⚪ Neutral: {neutral}")
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    # Reset playbook button
    if st.button("⚠️ Reset Playbook", use_container_width=True, type="secondary"):
        if os.path.exists("playbook.json"):
            os.remove("playbook.json")
        st.session_state.ace_crew.playbook = st.session_state.ace_crew.playbook.load("playbook.json")
        st.success("Playbook reset!")
        st.rerun()

# Main content area - Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "📚 Playbook", "📜 History", "📊 Analytics"])

# Tab 1: Query Interface
with tab1:
    st.subheader("Ask a Question")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **System Design:**
            - Should we prioritize speed or accuracy in recommendations?
            - How to handle distributed system failures?
            - What are database scaling strategies?
            """)
        with col2:
            st.markdown("""
            **ML/AI:**
            - How to prevent overfitting in neural networks?
            - When to use batch vs online learning?
            - Best practices for model deployment?
            """)
    
    # Query input
    user_query = st.text_area(
        "Enter your query:",
        height=120,
        placeholder="Type your question here...",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_button = st.button("🚀 Run ACE Cycle", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    # Process query
    if run_button and user_query.strip():
        with st.spinner("🔄 Running ACE cycle... (This may take a minute)"):
            try:
                result = st.session_state.ace_crew.run_ace_cycle(user_query)
                
                # Add to history
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': user_query,
                    'result': result
                })
                
                # Display results
                st.success("✅ ACE Cycle completed!")
                
                # Final answer - prominently displayed
                st.markdown("### 🎯 Final Answer")
                answer = result['generator_output'].get('final_answer', 'N/A')
                st.info(answer)
                
                # Show playbook growth
                st.markdown(f"**📚 Playbook now has:** {result['playbook_stats']['bullets']} bullets across {result['playbook_stats']['sections']} sections")
                
                # Details in expanders
                with st.expander("🧠 Reasoning Process", expanded=False):
                    reasoning = result['generator_output'].get('reasoning', [])
                    if isinstance(reasoning, list):
                        for i, step in enumerate(reasoning, 1):
                            st.markdown(f"**Step {i}:** {step}")
                    else:
                        st.write(reasoning)
                
                with st.expander("📎 Referenced Bullets"):
                    bullet_ids = result['generator_output'].get('bullet_ids', [])
                    if bullet_ids:
                        for bid in bullet_ids:
                            bullet = st.session_state.ace_crew.playbook.bullets.get(bid)
                            if bullet:
                                st.markdown(f"- **[{bid}]** {bullet.content}")
                    else:
                        st.write("No bullets referenced (playbook may be empty)")
                
                with st.expander("🔍 Reflection Analysis"):
                    ref_output = result['reflector_output']
                    st.markdown(f"**Error Identification:** {ref_output.get('error_identification', 'N/A')}")
                    st.markdown(f"**Root Cause:** {ref_output.get('root_cause_analysis', 'N/A')}")
                    st.markdown(f"**Correct Approach:** {ref_output.get('correct_approach', 'N/A')}")
                    st.markdown(f"**Key Insight:** {ref_output.get('key_insight', 'N/A')}")
                
                with st.expander("📝 Playbook Updates"):
                    cur_output = result['curator_output']
                    st.markdown(f"**Reasoning:** {cur_output.get('reasoning', 'N/A')}")
                    operations = cur_output.get('operations', [])
                    if operations:
                        for op in operations:
                            op_type = op.get('type', 'UNKNOWN')
                            if op_type == 'ADD':
                                st.success(f"➕ **ADD** to `{op.get('section', 'general')}`: {op.get('content', '')}")
                            elif op_type == 'UPDATE':
                                st.warning(f"✏️ **UPDATE** [{op.get('bullet_id', '')}]: {op.get('content', '')}")
                            elif op_type == 'REMOVE':
                                st.error(f"➖ **REMOVE**: {op.get('bullet_id', '')}")
                    else:
                        st.write("No updates made to playbook")
                
            except Exception as e:
                st.error(f"❌ Error during ACE cycle: {str(e)}")
                st.exception(e)
    elif run_button:
        st.warning("⚠️ Please enter a query first!")

# Tab 2: Playbook Viewer
with tab2:
    st.subheader("Current Playbook")
    
    playbook = st.session_state.ace_crew.playbook
    
    if not playbook.bullets:
        st.info("📭 Playbook is empty. Run some queries to build it up!")
    else:
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="stat-box"><h3>📚 Sections</h3><h2>{}</h2></div>'.format(
                len(playbook.sections)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-box"><h3>📝 Total Bullets</h3><h2>{}</h2></div>'.format(
                len(playbook.bullets)), unsafe_allow_html=True)
        with col3:
            helpful_count = sum(b.helpful for b in playbook.bullets.values())
            st.markdown('<div class="stat-box"><h3>✅ Helpful Tags</h3><h2>{}</h2></div>'.format(
                helpful_count), unsafe_allow_html=True)
        
        st.divider()
        
        # Filter by section
        sections = ["All"] + sorted(playbook.sections.keys())
        selected_section = st.selectbox("Filter by section:", sections)
        
        # Display by sections
        for section, bullet_ids in sorted(playbook.sections.items()):
            if selected_section != "All" and section != selected_section:
                continue
                
            st.markdown(f'<p class="section-header">📂 {section.upper()}</p>', unsafe_allow_html=True)
            
            for bullet_id in bullet_ids:
                bullet = playbook.bullets[bullet_id]
                
                # Color code based on tags
                if bullet.helpful > bullet.harmful:
                    border_color = "#2ca02c"  # green
                elif bullet.harmful > bullet.helpful:
                    border_color = "#d62728"  # red
                else:
                    border_color = "#1f77b4"  # blue
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 0.75rem; margin: 0.5rem 0; 
                            border-radius: 0.5rem; border-left: 4px solid {border_color};">
                    <strong>[{bullet.id}]</strong> {bullet.content}<br>
                    <small>✅ {bullet.helpful} | ❌ {bullet.harmful} | ⚪ {bullet.neutral} | 
                    Created: {bullet.created_at[:10]} | Updated: {bullet.updated_at[:10]}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Export playbook
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Export Playbook as JSON", use_container_width=True):
                playbook_json = json.dumps(playbook.dict(), indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=playbook_json,
                    file_name=f"playbook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        with col2:
            if st.button("📄 View Raw Playbook", use_container_width=True):
                st.code(playbook.as_prompt(), language="markdown")

# Tab 3: History
with tab3:
    st.subheader("Query History")
    
    if not st.session_state.history:
        st.info("📭 No queries yet. Start asking questions!")
    else:
        # Reverse chronological order
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"#{len(st.session_state.history) - i + 1} - {entry['query'][:80]}... ({entry['timestamp']})"):
                st.markdown(f"**🔍 Query:** {entry['query']}")
                st.markdown(f"**💡 Answer:** {entry['result']['generator_output'].get('final_answer', 'N/A')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bullets Used", len(entry['result']['generator_output'].get('bullet_ids', [])))
                with col2:
                    operations = len(entry['result']['curator_output'].get('operations', []))
                    st.metric("Playbook Updates", operations)
                with col3:
                    st.metric("Sections", entry['result']['playbook_stats']['sections'])
                
                st.markdown(f"**📝 Curator Reasoning:** {entry['result']['curator_output'].get('reasoning', 'N/A')}")

# Tab 4: Analytics
with tab4:
    st.subheader("System Analytics")
    
    playbook = st.session_state.ace_crew.playbook
    
    if not playbook.bullets:
        st.info("📊 No data yet. Run some queries first!")
    else:
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        helpful = sum(b.helpful for b in playbook.bullets.values())
        harmful = sum(b.harmful for b in playbook.bullets.values())
        neutral = sum(b.neutral for b in playbook.bullets.values())
        total_feedback = helpful + harmful + neutral
        
        with col1:
            st.metric("Total Feedback", total_feedback)
        with col2:
            st.metric("✅ Helpful", helpful)
        with col3:
            st.metric("❌ Harmful", harmful)
        with col4:
            st.metric("⚪ Neutral", neutral)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 Bullets by Section")
            section_data = {section: len(bullet_ids) for section, bullet_ids in playbook.sections.items()}
            for section, count in sorted(section_data.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- **{section}**: {count} bullets")
        
        with col2:
            st.markdown("### 📊 Feedback Distribution")
            if total_feedback > 0:
                helpful_pct = (helpful / total_feedback) * 100
                harmful_pct = (harmful / total_feedback) * 100
                neutral_pct = (neutral / total_feedback) * 100
                
                st.markdown(f"- ✅ Helpful: {helpful_pct:.1f}%")
                st.markdown(f"- ❌ Harmful: {harmful_pct:.1f}%")
                st.markdown(f"- ⚪ Neutral: {neutral_pct:.1f}%")
            else:
                st.write("No feedback yet")
        
        # Most helpful bullets
        st.divider()
        st.markdown("### 🌟 Top 10 Most Helpful Bullets")
        sorted_bullets = sorted(
            playbook.bullets.values(),
            key=lambda b: b.helpful,
            reverse=True
        )[:10]
        
        if any(b.helpful > 0 for b in sorted_bullets):
            for rank, bullet in enumerate(sorted_bullets, 1):
                if bullet.helpful > 0:
                    st.markdown(f"**{rank}. [{bullet.id}]** {bullet.content} (✅ {bullet.helpful})")
        else:
            st.write("No helpful bullets yet")
        
        # Most harmful bullets (to review)
        st.divider()
        st.markdown("### ⚠️ Bullets Needing Review (Most Harmful)")
        harmful_bullets = sorted(
            playbook.bullets.values(),
            key=lambda b: b.harmful,
            reverse=True
        )[:5]
        
        if any(b.harmful > 0 for b in harmful_bullets):
            for bullet in harmful_bullets:
                if bullet.harmful > 0:
                    st.markdown(f"**[{bullet.id}]** {bullet.content} (❌ {bullet.harmful})")
        else:
            st.write("No harmful bullets")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Agentic RAG System | Self-Improving Knowledge Base | Powered by CrewAI</small>
</div>
""", unsafe_allow_html=True)