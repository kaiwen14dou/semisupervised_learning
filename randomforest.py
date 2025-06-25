"""Random forest on different outcomes"""
cutoffs = {
    "hads_anxiety": 7,
    "hads_depression": 7,
    "dt": 4,
    "vfq": outcomes["vfq"].median()
}

ntree_values = [100, 200, 300, 400, 500]
n_folds = 5
results = [] 

# X must already be loaded and cleaned, same length as outcomes

for outcome_name, cutoff in cutoffs.items():
    print(f"\nOutcome: {outcome_name} (cutoff = {cutoff})")

    y = (outcomes[outcome_name] >= cutoff).astype(int).reset_index(drop=True)

    # Pie chart of class distribution
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=[f"Class {i} ({counts[i]})" for i in counts.index],
            autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
    plt.title(f"{outcome_name} | Outcome Distribution")
    plt.tight_layout()
    plt.show()

    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    roc_storage = []

    for ntree in ntree_values:
        print(f"\nntree = {ntree}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        cv_aucs = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            clf_cv = RandomForestClassifier(n_estimators=ntree, random_state=42)
            clf_cv.fit(X_cv_train, y_cv_train)
            y_cv_prob = clf_cv.predict_proba(X_cv_val)[:, 1]

            auc_val = roc_auc_score(y_cv_val, y_cv_prob)
            cv_aucs.append(auc_val)

            fpr, tpr, _ = roc_curve(y_cv_val, y_cv_prob)
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        mean_cv_auc = np.mean(cv_aucs)
        print(f"Mean CV AUC: {mean_cv_auc:.3f}")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(base_fpr, mean_tpr)

        roc_storage.append({
            "ntree": ntree,
            "fpr": base_fpr,
            "tpr": mean_tpr,
            "auc": mean_auc
        })

        # Final model
        clf_final = RandomForestClassifier(n_estimators=ntree, random_state=42)
        clf_final.fit(X_train, y_train)
        y_test_prob = clf_final.predict_proba(X_test)[:, 1]
        y_test_pred = clf_final.predict(X_test)

        test_auc = roc_auc_score(y_test, y_test_prob)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Test AUC: {test_auc:.3f} | Accuracy: {test_acc:.3f}")

        # Raw confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        class_names = ['Class 0', 'Class 1']

        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{outcome_name} | ntree = {ntree} | Confusion Matrix (Raw Counts)")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

        # ROC on test set
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {test_auc:.3f})", lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{outcome_name} | ntree = {ntree} | Test ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

        results.append({
            "outcome": outcome_name,
            "ntree": ntree,
            "mean_cv_auc": mean_cv_auc,
            "test_auc": test_auc,
            "test_accuracy": test_acc
        })

    # Plot mean CV ROC curves across ntrees
    plt.figure(figsize=(7, 6))
    for item in roc_storage:
        plt.plot(item["fpr"], item["tpr"],
                 label=f"ntree={item['ntree']} (AUC={item['auc']:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{outcome_name} | Mean CV ROC Curves by ntree")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Final result summary
results_df = pd.DataFrame(results)
print("\nSummary of All Results:")
print(results_df.round(3))