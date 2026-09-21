import streamlit as st
import pandas as pd
import json
import time
import os
from src.audit_log import get_pending_review, submit_feedback, get_all_logs, get_feedback_stats
from decision_pipeline import make_decision
from src.retraining_agent import run_agentic_retraining

st.set_page_config(page_title="AI Risk Manager — Fraud Review", layout="wide")
st.title("🛡️ AI Risk Manager — Human Review Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Pending Review", 
    "✅ Review Approved", 
    "📊 Audit Trail", 
    "⚙️ Simulate Transaction", 
    "💰 Cost Metrics", 
    "Model Versions"
])

# ============================================================================
# TAB 1: PENDING REVIEW (BLOCKED TRANSACTIONS)
# ============================================================================
with tab1:
    st.subheader("Transactions Awaiting Human Review")
    st.caption("Review blocked transactions. Mark 'Agree' if model was correct, 'Disagree' if it wrongly blocked.")
    
    pending = get_pending_review(limit=50)

    if not pending:
        st.info("✓ No pending transactions for review.")
    else:
        for item in pending:
            txn_data = json.loads(item['transaction_data'])
            
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Log ID:** {item['id']} | **Decision:** `{item['decision'].upper()}` "
                                f"| **Risk Score:** {item['risk_score']:.4f}")
                    st.markdown(f"**Transaction:** ₹{txn_data.get('amount', 0):,.2f} "
                                f"({txn_data.get('type', 'N/A')})")
                    
                    # Ground truth (testing only)
                    true_label = txn_data.get('_true_label', 'N/A')
                    if true_label != 'N/A':
                        label_text = "🔴 ACTUALLY FRAUD" if true_label == 1 else "🟢 ACTUALLY LEGIT"
                        st.caption(f"**[TESTING ONLY] Ground truth:** {label_text}")
                    
                    st.markdown(f"**Why flagged:** {item['explanation']}")
                    st.caption(f"Timestamp: {item['timestamp']}")
                
                with col2:
                    # AGREE BUTTON
                    if st.button("✅ Agree", key=f"agree_{item['id']}", use_container_width=True):
                        submit_feedback(item['id'], 'agree')
                        st.success("✓ Feedback recorded")
                        time.sleep(0.5)
                        st.rerun()
                    
                    # DISAGREE BUTTON
                    if st.button("❌ Disagree\nActually Legit", key=f"disagree_{item['id']}", use_container_width=True):
                        submit_feedback(item['id'], 'disagree_should_approve')
                        st.error("✗ Feedback recorded")
                        time.sleep(0.5)
                        st.rerun()


# ============================================================================
# TAB 2: REVIEW APPROVED (SPOT-CHECK FOR FALSE NEGATIVES)
# ============================================================================
with tab2:
    st.subheader("Spot-Check Approved Transactions")
    st.caption("Review auto-approved transactions to catch missed fraud (false negatives).")
    
    all_logs = get_all_logs(limit=500)
    approved_unreviewed = [
        log for log in all_logs 
        if log['decision'] == 'approve' and log['human_reviewed'] == 0
    ]

    if not approved_unreviewed:
        st.info("✓ No unreviewed approved transactions.")
    else:
        for item in approved_unreviewed:
            txn_data = json.loads(item['transaction_data'])
            
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Log ID:** {item['id']} | **Decision:** `APPROVE` "
                                f"| **Risk Score:** {item['risk_score']:.4f}")
                    st.markdown(f"**Transaction:** ₹{txn_data.get('amount', 0):,.2f} "
                                f"({txn_data.get('type', 'N/A')})")
                    
                    # Ground truth (testing only)
                    true_label = txn_data.get('_true_label', 'N/A')
                    if true_label != 'N/A':
                        label_text = "🔴 ACTUALLY FRAUD" if true_label == 1 else "🟢 ACTUALLY LEGIT"
                        st.caption(f"**[TESTING ONLY] Ground truth:** {label_text}")
                    
                    st.caption(f"Timestamp: {item['timestamp']}")
                
                with col2:
                    # AGREE BUTTON
                    if st.button("✅ Agree\nLegit", key=f"appr_agree_{item['id']}", use_container_width=True):
                        submit_feedback(item['id'], 'agree')
                        st.success("✓ Feedback recorded")
                        time.sleep(0.5)
                        st.rerun()
                    
                    # DISAGREE BUTTON
                    if st.button("🚨 Disagree\nShould Block", key=f"appr_disagree_{item['id']}", use_container_width=True):
                        submit_feedback(item['id'], 'disagree_should_block')
                        st.error("✗ Feedback recorded")
                        time.sleep(0.5)
                        st.rerun()


# ============================================================================
# TAB 3: AUDIT TRAIL & AUTONOMOUS RETRAINING
# ============================================================================
with tab3:
    st.subheader("Audit Trail & Autonomous Retraining")
    
    # Get feedback count
    all_logs = get_all_logs(limit=1000)
    reviewed_count = len([l for l in all_logs if l['human_reviewed'] == 1])
    
    # === SUBMIT FEEDBACK & RETRAIN BUTTON ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📤 Submit Feedback & Retrain Model", type="primary", use_container_width=True):
            if reviewed_count == 0:
                st.error("❌ No feedback submitted yet. Review transactions in Tabs 1-2 first.")
            else:
                with st.spinner(f"🤖 Agent reasoning over {reviewed_count} feedback rows..."):
                    result = run_agentic_retraining(feedback_rows=reviewed_count)
                    st.session_state['last_retrain_result'] = result
                    time.sleep(1)
    
    with col2:
        st.metric("Feedback Rows", reviewed_count)
    
    st.divider()
    
    # === DISPLAY AGENT RESULTS ===
    if 'last_retrain_result' in st.session_state:
        result = st.session_state['last_retrain_result']
        
        if result.get('error'):
            st.error(f"⚠️ Agent Error: {result.get('error')}")
        else:
            decision = result.get('final_decision', 'unknown')
            
            if decision == 'promote':
                st.success(f"✅ **AGENT DECISION: PROMOTE to v{result.get('version_number')}**")
                st.write(f"**Confidence:** {result.get('confidence', 'unknown').upper()}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current F1", f"{result.get('current_f1', 0):.3f}")
                with col2:
                    delta_f1 = (result.get('candidate_f1', 0) - result.get('current_f1', 0))
                    st.metric("Candidate F1", f"{result.get('candidate_f1', 0):.3f}", 
                             delta=f"+{delta_f1:.3f}")
                with col3:
                    st.metric("Current Recall", f"{result.get('current_recall', 0):.3f}")
                with col4:
                    delta_recall = (result.get('candidate_recall', 0) - result.get('current_recall', 0))
                    st.metric("Candidate Recall", f"{result.get('candidate_recall', 0):.3f}",
                             delta=f"+{delta_recall:.3f}")
                
            elif decision == 'reject':
                st.error(f"❌ **AGENT DECISION: REJECT**")
                st.write("Candidate model did not meet promotion criteria. Current model retained.")
                st.write(f"**Confidence:** {result.get('confidence', 'unknown').upper()}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current F1", f"{result.get('current_f1', 0):.3f}")
                with col2:
                    delta_f1 = (result.get('candidate_f1', 0) - result.get('current_f1', 0))
                    st.metric("Candidate F1", f"{result.get('candidate_f1', 0):.3f}",
                             delta=f"{delta_f1:.3f}")
                with col3:
                    st.metric("Current Recall", f"{result.get('current_recall', 0):.3f}")
                with col4:
                    delta_recall = (result.get('candidate_recall', 0) - result.get('current_recall', 0))
                    st.metric("Candidate Recall", f"{result.get('candidate_recall', 0):.3f}",
                             delta=f"{delta_recall:.3f}")
            
            with st.expander("🤖 Agent Reasoning & Steps"):
                st.write(result.get('final_reasoning', 'No reasoning provided'))
                st.caption(f"Agent took {len(result.get('steps', []))} reasoning steps")
                st.caption(f"Timestamp: {result.get('timestamp')}")
    
    st.divider()
    
    # === FULL DECISION LOG TABLE ===
    st.markdown("### Full Decision Log")
    logs = get_all_logs(limit=200)
    
    if logs:
        display_data = []
        for log in logs:
            txn = json.loads(log['transaction_data'])
            display_data.append({
                'id': log['id'],
                'amount': f"₹{txn.get('amount', 0):,.0f}",
                'type': txn.get('type', 'N/A'),
                'decision': log['decision'].upper(),
                'risk_score': f"{log['risk_score']:.3f}",
                'reviewed': '✓' if log['human_reviewed'] else '',
                'feedback': log['human_feedback'] or '-'
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Feedback stats
        st.markdown("#### Feedback Statistics")
        reviewed_logs = [l for l in logs if l['human_reviewed'] == 1]
        if reviewed_logs:
            agree_count = len([l for l in reviewed_logs if l['human_feedback'] == 'agree'])
            disagree_approve = len([l for l in reviewed_logs if l['human_feedback'] == 'disagree_should_approve'])
            disagree_block = len([l for l in reviewed_logs if l['human_feedback'] == 'disagree_should_block'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviewed", len(reviewed_logs))
            col2.metric("Agree", agree_count)
            col3.metric("Disagree (→Approve)", disagree_approve)
            col4.metric("Disagree (→Block)", disagree_block)
        else:
            st.info("No feedback submitted yet.")
    else:
        st.info("No transactions logged yet.")


# ============================================================================
# TAB 4: SIMULATE TRANSACTION
# ============================================================================
with tab4:
    st.subheader("Simulate a New Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=10000.0)
        txn_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT"])
        old_orig = st.number_input("Sender's balance before", min_value=0.0, value=10000.0)
        new_orig = st.number_input("Sender's balance after", min_value=0.0, value=0.0)
    
    with col2:
        old_dest = st.number_input("Destination's balance before", min_value=0.0, value=0.0)
        new_dest = st.number_input("Destination's balance after", min_value=0.0, value=10000.0)

    if st.button("Run Detection"):
        txn = {
            'amount': amount,
            'type': txn_type,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest
        }
        
        result = make_decision(txn)
        
        color = {'approve': 'green', 'hold': 'orange', 'block': 'red'}[result['decision']]
        st.markdown(f"### Decision: :{color}[{result['decision'].upper()}]")
        st.markdown(f"**Risk Score:** {result['risk_score']:.4f}")
        st.markdown(f"**Explanation:** {result['explanation']}")
        st.caption(f"Logged as ID {result['log_id']}")


# ============================================================================
# TAB 5: COST METRICS
# ============================================================================
with tab5:
    st.subheader("Model Evaluation — Cost & Error Analysis")

    st.markdown("### Validated Test-Set Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", "92.6%", help="% of actual fraud caught")
    col2.metric("Missed Fraud Cost", "$113.1M", help="Total value of fraud not caught")
    col3.metric("False Block Cost", "$75.3K", help="Friction cost of wrongly blocking legit txns")

    st.divider()

    st.markdown("### Live Session Metrics")
    logs = get_all_logs(limit=1000)

    if not logs:
        st.info("No transactions processed yet.")
    else:
        df = pd.DataFrame(logs)
        df['transaction_data'] = df['transaction_data'].apply(lambda x: json.loads(x))
        df['amount'] = df['transaction_data'].apply(lambda x: x.get('amount', 0))

        total_txns = len(df)
        blocked = (df['decision'] == 'block').sum()
        approved = (df['decision'] == 'approve').sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", total_txns)
        col2.metric("Blocked", blocked)
        col3.metric("Approved", approved)

        st.markdown("#### Error Breakdown (from human feedback)")
        reviewed = df[df['human_reviewed'] == 1]

        if reviewed.empty:
            st.info("No feedback yet — review transactions to populate analysis.")
        else:
            false_positives = reviewed[reviewed['human_feedback'] == 'disagree_should_approve']
            false_negatives = reviewed[reviewed['human_feedback'] == 'disagree_should_block']
            confirmed_correct = reviewed[reviewed['human_feedback'] == 'agree']

            fp_cost = len(false_positives) * 5.0
            fn_cost = false_negatives['amount'].sum() if len(false_negatives) > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Confirmed Correct", len(confirmed_correct))
            col2.metric("False Positives", len(false_positives))
            col3.metric("False Negatives", len(false_negatives))

            st.markdown(f"**Estimated cost:** ${fp_cost + fn_cost:,.2f} "
                        f"(False-block: ${fp_cost:,.2f} | Missed fraud: ${fn_cost:,.2f})")

        st.divider()
        st.markdown("### Fraud Detection Trend")
        
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df['is_flagged'] = (df['decision'] == 'block').astype(int)
        trend = df.set_index('timestamp_dt').resample('1min')['is_flagged'].sum()

        st.line_chart(trend)

        if len(trend) >= 3:
            recent = trend.iloc[-1]
            avg = trend.iloc[:-1].mean()
            if avg > 0 and recent > avg * 2:
                st.error(f"⚠️ SPIKE: {recent} flagged txns vs avg {avg:.1f}")


# ============================================================================
# TAB 6: MODEL VERSIONS & DECISION LOG
# ============================================================================
with tab6:
    st.subheader("Model Version History & Retraining Decisions")

    if os.path.exists('models/version_registry.json'):
        with open('models/version_registry.json') as f:
            registry = json.load(f)

        st.metric("Current Live Version", f"v{registry.get('current_version', 0)}")

        st.markdown("### Retraining Decision Log")
        versions = registry.get('versions', [])
        
        if versions:
            for v in reversed(versions):
                decision_color = "🟢" if v.get('promoted') else "🔴"
                version_label = f"v{v['version']}" if v.get('version') else "Candidate (rejected)"
                
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"{decision_color} **{version_label}**")
                        st.caption(f"{v.get('timestamp', '')[:19]}")
                    
                    with col2:
                        st.write(f"**Status:** {'✅ PROMOTED' if v.get('promoted') else '❌ REJECTED'}")
                        st.write(f"**Feedback rows:** {v.get('feedback_rows', 0)}")
                        st.write(f"**Feedback quality:** {v.get('feedback_quality', 'N/A')}")
                    
                    with st.expander("Metrics & Agent Reasoning"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Current Model**")
                            st.write(f"F1: {(v.get('current_f1') or 0):.3f}")
                            st.write(f"Recall: {(v.get('current_recall') or 0):.3f}")
                        
                        with col2:
                            st.write("**Candidate Model**")
                            st.write(f"F1: {(v.get('candidate_f1') or 0):.3f}")
                            st.write(f"Recall: {(v.get('candidate_recall') or 0):.3f}")
                        
                        st.write("**Agent Decision**")
                        decision_val = v.get('decision') or 'N/A'
                        confidence_val = v.get('confidence') or 'N/A'
                        st.write(f"Decision: {decision_val.upper() if decision_val != 'N/A' else 'N/A'}")
                        st.write(f"Confidence: {confidence_val.upper() if confidence_val != 'N/A' else 'N/A'}")
                        
                        st.write("**Agent Reasoning**")
                        st.write(v.get('agent_reasoning', 'N/A'))
                        st.caption(f"Agent steps: {v.get('agent_steps', 0)}")
        else:
            st.info("No retraining history yet.")
    else:
        st.info("No model versions recorded yet. Submit feedback and retrain to create versions.")