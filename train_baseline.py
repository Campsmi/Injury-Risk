import joblib
from src.data_prep import *
from src.modeling import *
from src.evaluation import *


def main():
    
    dataset = load_dataset("data/collegiate_athlete_injury_dataset.csv")
    X, y = prepare_feature(dataset, "Injury_Indicator")
    X_train, X_test, Y_train, Y_test = create_splits(X, y, test_size = 0.3, random_state = 42)
    
    cat_cols, num_cols = get_feature_groups(X)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    model_builders = {
    "random_forest": build_random_forest(),
    "logistic_regression": build_logistic_regression(),
    "gradient_boosting": build_gradient_boosting(),
    "mlp": build_mlp(),
    }

    for model_name, builder_fn in model_builders.items():
        print(f"\n=== Training {model_name} ===")
        
        model = builder_fn
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, Y_train)

        roc_auc, accuracy = evaluate_model(pipeline, X_test, Y_test)

        print(f"{model_name} ROC-AUC: {roc_auc:.3f}")
        
        # Save
        save_path = f"models/{model_name}.joblib"
        joblib.dump(pipeline, save_path)
        print(f"Saved {model_name} → {save_path}")
        print(f"ROC-AUC for {model_name}: {roc_auc:.3f}")
        print(f"Accuracy for {model_name}: {accuracy:.3f}")
        
        
        
        
if __name__ == "__main__":
    main()