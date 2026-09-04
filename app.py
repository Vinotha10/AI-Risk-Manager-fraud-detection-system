import streamlit as st
import pandas as pd
import json
from src.audit_log import get_pending_review, submit_feedback, get_all_logs, get_feedback_stats
from decision_pipeline import make_decision, THRESHOLDS

st.set_page_config(page_title="AI Risk Manager — Fraud Review", layout="wide")

st.title("🛡️ AI Risk Manager — Human Review Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Pending Review", "✅ Review Approved", "📊 Audit Trail", 
    "⚙️ Simulate Transaction", "💰 Cost Metrics", "Model Versions"
])
# ---- TAB 1: Pending Review ----
with tab1:
    st.subheader("Transactions Awaiting Human Review")
    pending = get_pending_review(limit=50)

    if not pending:
        st.info("No transactions currently pending review.")
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
                    true_label = txn_data.get('_true_label', 'N/A')
                    label_text = "🔴 ACTUALLY FRAUD" if true_label == 1 else ("🟢 ACTUALLY LEGIT" if true_label == 0 else "N/A")
                    st.caption(f"**[TESTING ONLY] Ground truth:** {label_text}")
                    st.markdown(f"**Why flagged:** {item['explanation']}")
                    st.caption(f"Reason: {item['decision_reason']} | Timestamp: {item['timestamp']}")

                with col2:
                    if st.button("✅ Agree", key=f"agree_{item['id']}"):
                        submit_feedback(item['id'], 'agree')
                        st.rerun()
                    if st.button("❌ Disagree — Actually Legit", key=f"disagree_legit_{item['id']}"):
                        submit_feedback(item['id'], 'disagree_should_approve')
                        st.rerun()
                    if item['decision'] != 'block':
                        if st.button("🚨 Disagree — Should Block", key=f"disagree_block_{item['id']}"):
                            submit_feedback(item['id'], 'disagree_should_block')
                            st.rerun()

# ---- TAB 2: Review Approved Transactions ----
with tab2:
    st.subheader("Spot-Check Approved Transactions")
    st.caption("Review transactions the model auto-approved, to catch any missed fraud (false negatives).")

    from src.audit_log import get_all_logs

    all_logs = get_all_logs(limit=500)
    approved_unreviewed = [
        log for log in all_logs 
        if log['decision'] == 'approve' and log['human_reviewed'] == 0
    ]

    if not approved_unreviewed:
        st.info("No unreviewed approved transactions.")
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

                    true_label = txn_data.get('_true_label', 'N/A')
                    label_text = "🔴 ACTUALLY FRAUD" if true_label == 1 else ("🟢 ACTUALLY LEGIT" if true_label == 0 else "N/A")
                    st.caption(f"**[TESTING ONLY] Ground truth:** {label_text}")
                    st.caption(f"Timestamp: {item['timestamp']}")

                with col2:
                    if st.button("✅ Agree — Correctly Approved", key=f"appr_agree_{item['id']}"):
                        submit_feedback(item['id'], 'agree')
                        st.rerun()
                    if st.button("🚨 Disagree — Should Block", key=f"appr_disagree_{item['id']}"):
                        submit_feedback(item['id'], 'disagree_should_block')
                        st.rerun()


# ---- TAB 3: Audit Trail ----
with tab3:
    st.subheader("Full Decision Log")
    logs = get_all_logs(limit=200)
    if logs:
        df = pd.DataFrame(logs)
        df['transaction_data'] = df['transaction_data'].apply(lambda x: json.loads(x))
        df['amount'] = df['transaction_data'].apply(lambda x: x.get('amount', None))
        df['type'] = df['transaction_data'].apply(lambda x: x.get('type', None))

        display_cols = ['id', 'timestamp', 'amount', 'type', 'risk_score',
                         'decision', 'decision_reason', 'human_reviewed', 'human_feedback']
        st.dataframe(df[display_cols], use_container_width=True)

        st.subheader("Feedback Statistics")
        stats = get_feedback_stats()
        if stats:
            stats_df = pd.DataFrame(stats, columns=['Decision', 'Human Feedback', 'Count'])
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No feedback submitted yet.")
    else:
        st.info("No logs yet. Simulate a transaction in the third tab.")

    st.subheader("Full Decision Log")
    
    # --- Retrain button ---
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Retrain Model", type="primary"):
            with st.spinner("Running guarded retraining cycle..."):
                from retrain_pipeline import run_retraining_cycle
                result = run_retraining_cycle()
            
            if result['status'] == 'no_feedback' if 'status' in result else False:
                st.warning("No feedback available yet.")
            else:
                st.session_state['last_retrain_result'] = result
    
    # Show last retrain result if available
    if 'last_retrain_result' in st.session_state:
        result = st.session_state['last_retrain_result']
        if result.get('promoted'):
            st.success(f"✅ Model promoted! F1: {result['current_scores']['f1']:.3f} → "
                       f"{result['candidate_scores']['f1']:.3f}")
            st.success(f"✅ Model promoted! ...")
            # Force reload of the model in explain.py's module-level state
            import importlib
            import src.explain
            importlib.reload(src.explain)
            st.info("Live model reloaded — new decisions will use the promoted model.")
        else:
            st.error(f"❌ Candidate rejected. F1: {result['current_scores']['f1']:.3f} → "
                     f"{result['candidate_scores']['f1']:.3f} (regression, kept existing model)")
        
        with st.expander("Full retrain details"):
            st.json(result)

    st.divider()
    
    logs = get_all_logs(limit=200)
    # ... rest of existing audit trail table code ...

# ---- TAB 4: Simulate Transaction ----
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
            'amount': amount, 'type': txn_type,
            'oldbalanceOrg': old_orig, 'newbalanceOrig': new_orig,
            'oldbalanceDest': old_dest, 'newbalanceDest': new_dest
        }
        result = make_decision(txn)

        color = {'approve': 'green', 'hold': 'orange', 'block': 'red'}[result['decision']]
        st.markdown(f"### Decision: :{color}[{result['decision'].upper()}]")
        st.markdown(f"**Risk Score:** {result['risk_score']:.4f}")
        st.markdown(f"**Explanation:** {result['explanation']}")
        st.caption(f"Logged as ID {result['log_id']}")


# ---- TAB 5: Cost Metrics ----
with tab5:
    st.subheader("Model Evaluation — Cost & Error Analysis")

    st.markdown("### Validated Test-Set Performance (552,504 held-out transactions)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", "92.6%", help="% of actual fraud correctly caught")
    col2.metric("Missed Fraud Cost", "$113.1M", help="Total value of fraud not caught (false negatives)")
    col3.metric("False Block Cost", "$75,305", help="Estimated friction cost of wrongly blocking legit transactions")

    st.divider()

    st.markdown("### Live Session Metrics (from current audit trail)")
    logs = get_all_logs(limit=1000)

    if not logs:
        st.info("No transactions processed yet. Run some in the Simulate tab or batch simulation.")
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

        st.markdown("#### False Positive / False Negative Breakdown (where feedback is available)")
        reviewed = df[df['human_reviewed'] == 1]

        if reviewed.empty:
            st.info("No human feedback yet — review some Pending items to populate this analysis.")
        else:
            false_positives = reviewed[reviewed['human_feedback'] == 'disagree_should_approve']
            false_negatives = reviewed[reviewed['human_feedback'] == 'disagree_should_block']
            confirmed_correct = reviewed[reviewed['human_feedback'] == 'agree']

            fp_cost = len(false_positives) * 5.0    # matches your $5 false-block cost assumption
            fn_cost = false_negatives['amount'].sum()  # missed fraud = real amount lost

            col1, col2, col3 = st.columns(3)
            col1.metric("Confirmed Correct", len(confirmed_correct))
            col2.metric("False Positives (wrongly blocked)", len(false_positives),
                        help=f"Estimated cost: ${fp_cost:,.2f}")
            col3.metric("False Negatives (missed fraud)", len(false_negatives),
                        help=f"Estimated cost: ${fn_cost:,.2f}")

            st.markdown(f"**Estimated cost from reviewed cases:** "
                        f"${fp_cost + fn_cost:,.2f} "
                        f"(False block cost: ${fp_cost:,.2f} | Missed fraud cost: ${fn_cost:,.2f})")

        st.markdown("#### Full Reviewed Cases")
        if not reviewed.empty:
            display_df = reviewed[['id', 'amount', 'decision', 'risk_score', 'human_feedback']].copy()
            st.dataframe(display_df, use_container_width=True)

        st.divider()
    st.markdown("### Fraud Detection Trend Over Time")

    if not df.empty:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df['is_flagged'] = df['decision'] == 'block'

        # Bucket into time windows (adjust freq based on how spread your timestamps are)
        trend = df.set_index('timestamp_dt').resample('1min')['is_flagged'].sum()

        st.line_chart(trend)

        # Simple spike detection: current window vs rolling average
        if len(trend) >= 3:
            recent = trend.iloc[-1]
            avg = trend.iloc[:-1].mean()
            if avg > 0 and recent > avg * 2:
                st.error(f"⚠️ FRAUD SPIKE ALERT: {recent} flagged transactions in the last window "
                         f"vs average of {avg:.1f} — investigate immediately.")

with tab6:  # add this as a new tab
    st.subheader("Model Version History")

    import os
    if os.path.exists('models/version_registry.json'):
        with open('models/version_registry.json') as f:
            registry = json.load(f)

        st.metric("Current Live Version", f"v{registry['current_version']}")

        st.markdown("### Version Log")
        for v in reversed(registry['versions']):
            status = "✅ PROMOTED" if v['promoted'] else "❌ REJECTED"
            version_label = f"v{v['version']}" if v['version'] else "candidate (rejected)"
            with st.container(border=True):
                st.markdown(f"**{version_label}** — {status}")
                st.caption(f"Timestamp: {v['timestamp']} | Feedback rows used: {v['feedback_rows_used']}")
                st.write(f"F1: {v['scores']['f1']:.3f} | Precision: {v['scores']['precision']:.3f} | Recall: {v['scores']['recall']:.3f}")
    else:
        st.info("No retraining cycles run yet.")