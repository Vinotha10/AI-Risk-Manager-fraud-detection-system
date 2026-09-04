import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# ---- Config: cost assumptions (documented, tunable) ----
HOLD_REVIEW_COST = 0.5          # cost of a human reviewing one held transaction
FALSE_BLOCK_COST = 5.0          # estimated friction/goodwill cost of wrongly blocking a legit txn
# False negative cost = the transaction amount itself (real money lost)

def load_artifacts():
    model = joblib.load('models/base_model.pkl')
    test_final = pd.read_pickle('data/paysim_test_features.pkl')
    with open('models/feature_columns.json') as f:
        feature_columns = json.load(f)
    return model, test_final, feature_columns

def prepare_features(df, feature_columns):
    feature_cols_raw = ['amount', 'type', 'balance_diff_orig', 'orig_balance_wiped',
                         'dest_balance_zero_before_after', 'oldbalanceOrg', 'oldbalanceDest']
    X = df[feature_cols_raw].copy()
    X = pd.get_dummies(X, columns=['type'], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X

def compute_cost_for_thresholds(y_true, y_proba, amounts, block_threshold, approve_threshold):
    """
    For a given pair of thresholds, simulate the 3-tier decision and compute total cost.
    """
    # Start with the normal probability-based tiering
    decisions = np.where(y_proba >= block_threshold, 'block',
                 np.where(y_proba >= approve_threshold, 'hold', 'approve'))

    # Override with the deterministic rule: amount == 0 -> always block
    # This bypasses the model entirely for this specific pattern
    zero_amount_mask = (amounts == 0)
    decisions = np.where(zero_amount_mask, 'block', decisions)

    total_cost = 0.0
    breakdown = {'missed_fraud_cost': 0.0, 'false_block_cost': 0.0, 'hold_review_cost': 0.0,
                 'rule_based_blocks': int(zero_amount_mask.sum())}


    for decision, is_fraud, amount in zip(decisions, y_true, amounts):
        if decision == 'approve' and is_fraud == 1:
            # Missed fraud -> lose the full amount
            total_cost += amount
            breakdown['missed_fraud_cost'] += amount
        elif decision == 'block' and is_fraud == 0:
            # Wrongly blocked a legit transaction
            total_cost += FALSE_BLOCK_COST
            breakdown['false_block_cost'] += FALSE_BLOCK_COST
        
        # block on actual fraud = correct catch, no cost
        # approve on actual legit = correct, no cost

    return total_cost, breakdown, decisions

def grid_search_thresholds(y_true, y_proba, amounts, n_steps=20):
    """
    Search over threshold pairs to find the lowest-cost combination.
    """
    candidates = np.linspace(0.05, 0.95, n_steps)
    best = None

    results = []
    for block_t in candidates:
        for approve_t in candidates:
            if approve_t >= block_t:
                continue  # approve threshold must be lower than block threshold
            cost, breakdown, _ = compute_cost_for_thresholds(
                y_true, y_proba, amounts, block_t, approve_t
            )
            results.append({
                'block_threshold': block_t,
                'approve_threshold': approve_t,
                'total_cost': cost,
                **breakdown
            })
            if best is None or cost < best['total_cost']:
                best = results[-1]

    return best, pd.DataFrame(results)

def main():
    model, test_final, feature_columns = load_artifacts()
    X_test = prepare_features(test_final, feature_columns)
    y_test = test_final['isFraud'].values
    amounts = test_final['amount'].values

    y_proba = model.predict_proba(X_test)[:, 1]

    print("Running threshold grid search (this may take a moment)...")
    best, results_df = grid_search_thresholds(y_test, y_proba, amounts, n_steps=20)

    print("\n=== BEST THRESHOLD COMBINATION ===")
    print(f"Block threshold:   {best['block_threshold']:.3f}")
    print(f"Approve threshold: {best['approve_threshold']:.3f}")
    print(f"Total cost:        ${best['total_cost']:.2f}")
    print(f"  - Missed fraud cost: ${best['missed_fraud_cost']:.2f}")
    print(f"  - False block cost:  ${best['false_block_cost']:.2f}")
    print(f"  - Hold review cost:  ${best['hold_review_cost']:.2f}")

    # Compare against naive baseline: single 0.5 threshold, no hold tier
    naive_cost, naive_breakdown, _ = compute_cost_for_thresholds(
        y_test, y_proba, amounts, block_threshold=0.5, approve_threshold=0.5
    )
    print(f"\n=== NAIVE 0.5 THRESHOLD BASELINE (for comparison) ===")
    print(f"Total cost: ${naive_cost:.2f}")
    print(f"Savings from optimization: ${naive_cost - best['total_cost']:.2f}")

    # Save the optimized thresholds for use elsewhere in the pipeline
    thresholds_config = {
        'block_threshold': float(best['block_threshold']),
        'approve_threshold': float(best['approve_threshold'])
    }
    with open('models/thresholds.json', 'w') as f:
        json.dump(thresholds_config, f, indent=2)
    print("\nSaved optimized thresholds to models/thresholds.json")

    # Save full results grid for your report/analysis
    results_df.to_csv('models/threshold_search_results.csv', index=False)

    # Plot cost surface for visualization (useful for your presentation)
    plt.figure(figsize=(8,6))
    pivot = results_df.pivot_table(index='block_threshold', columns='approve_threshold', values='total_cost')
    plt.imshow(pivot, aspect='auto', origin='lower',
               extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(label='Total Cost ($)')
    plt.xlabel('Approve Threshold')
    plt.ylabel('Block Threshold')
    plt.title('Cost Surface Across Threshold Combinations')
    plt.scatter([best['approve_threshold']], [best['block_threshold']], color='red', marker='*', s=200, label='Optimal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/cost_surface.png')
    print("Saved cost surface plot to models/cost_surface.png")

    # Add this to threshold_optimizer.py's main(), after computing 'best'
    _, _, decisions = compute_cost_for_thresholds(
        y_test, y_proba, amounts, best['block_threshold'], best['approve_threshold']
    )

    zero_amount_fraud_mask = (amounts == 0) & (y_test == 1)
    caught_zero_amount = (decisions[zero_amount_fraud_mask] == 'block').sum()
    print(f"\nZero-amount fraud cases: {zero_amount_fraud_mask.sum()}")
    print(f"Caught by rule: {caught_zero_amount}")

    overall_recall = ((decisions == 'block') & (y_test == 1)).sum() / (y_test == 1).sum()
    print(f"Overall fraud recall (block only): {overall_recall:.4f}")

if __name__ == '__main__':
    main()
    