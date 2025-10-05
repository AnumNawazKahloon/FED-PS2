"""
Example usage of visualization module
Run this after training models to generate all charts
"""

from visualization import create_visualization_report, ResultsVisualizer
from data_loader import load_data
from models import create_model
import torch

def example_basic_usage():
    """
    Example 1: Basic usage with just results dictionary
    (No interpretability analysis)
    """
    # Mock results (replace with actual results from training)
    results = {
        'LSTM': {
            'Centralized': {
                'mse': 0.012345,
                'mae': 0.087654,
                'rmse': 0.111111,
                'training_time': 45.23
            },
            'Federated': {
                'mse': 0.014567,
                'mae': 0.091234,
                'rmse': 0.120678,
                'training_time': 52.10
            }
        },
        'BiLSTM': {
            'Centralized': {
                'mse': 0.011234,
                'mae': 0.084321,
                'rmse': 0.105987,
                'training_time': 67.45
            },
            'Federated': {
                'mse': 0.013456,
                'mae': 0.088765,
                'rmse': 0.116012,
                'training_time': 74.32
            }
        },
        'CNN-LSTM': {
            'Centralized': {
                'mse': 0.010987,
                'mae': 0.082456,
                'rmse': 0.104789,
                'training_time': 89.67
            },
            'Federated': {
                'mse': 0.012876,
                'mae': 0.086543,
                'rmse': 0.113478,
                'training_time': 96.45
            }
        }
    }
    
    # Generate visualizations (without interpretability)
    print("Generating basic visualizations...")
    create_visualization_report(
        results=results,
        models_dict=None,  # No models provided
        test_data=None,    # No test data provided
        output_dir='./plots_basic/'
    )
    print("✓ Basic visualizations completed!")


def example_full_usage():
    """
    Example 2: Full usage with interpretability analysis
    (Includes feature importance and SHAP)
    """
    # Load data
    print("Loading data...")
    train_data, test_data, scaler = load_data()
    
    # Create and train models (simplified example)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Creating models...")
    lstm_model = create_model('lstm').to(device)
    bilstm_model = create_model('bilstm').to(device)
    cnnlstm_model = create_model('cnn-lstm').to(device)
    
    # Note: In real usage, these models would be trained
    # Here we're just showing the structure
    
    models_dict = {
        'LSTM': lstm_model,
        'BiLSTM': bilstm_model,
        'CNN-LSTM': cnnlstm_model
    }
    
    # Results from training (replace with actual results)
    results = {
        'LSTM': {
            'Centralized': {
                'mse': 0.012345,
                'mae': 0.087654,
                'rmse': 0.111111,
                'training_time': 45.23
            },
            'Federated': {
                'mse': 0.014567,
                'mae': 0.091234,
                'rmse': 0.120678,
                'training_time': 52.10
            }
        },
        'BiLSTM': {
            'Centralized': {
                'mse': 0.011234,
                'mae': 0.084321,
                'rmse': 0.105987,
                'training_time': 67.45
            },
            'Federated': {
                'mse': 0.013456,
                'mae': 0.088765,
                'rmse': 0.116012,
                'training_time': 74.32
            }
        },
        'CNN-LSTM': {
            'Centralized': {
                'mse': 0.010987,
                'mae': 0.082456,
                'rmse': 0.104789,
                'training_time': 89.67
            },
            'Federated': {
                'mse': 0.012876,
                'mae': 0.086543,
                'rmse': 0.113478,
                'training_time': 96.45
            }
        }
    }
    
    # Generate all visualizations including interpretability
    print("\nGenerating full visualizations with interpretability...")
    create_visualization_report(
        results=results,
        models_dict=models_dict,
        test_data=test_data,
        output_dir='./plots_full/'
    )
    print("✓ Full visualizations completed!")


def example_custom_visualization():
    """
    Example 3: Custom visualization using ResultsVisualizer class
    (Generate only specific charts)
    """
    results = {
        'LSTM': {
            'Centralized': {'mse': 0.012, 'mae': 0.088, 'rmse': 0.111, 'training_time': 45.2},
            'Federated': {'mse': 0.015, 'mae': 0.092, 'rmse': 0.122, 'training_time': 52.1}
        },
        'BiLSTM': {
            'Centralized': {'mse': 0.011, 'mae': 0.084, 'rmse': 0.106, 'training_time': 67.5},
            'Federated': {'mse': 0.013, 'mae': 0.089, 'rmse': 0.116, 'training_time': 74.3}
        },
        'CNN-LSTM': {
            'Centralized': {'mse': 0.011, 'mae': 0.082, 'rmse': 0.105, 'training_time': 89.7},
            'Federated': {'mse': 0.013, 'mae': 0.087, 'rmse': 0.113, 'training_time': 96.5}
        }
    }
    
    # Create visualizer
    visualizer = ResultsVisualizer(results, output_dir='./plots_custom/')
    
    # Generate only specific plots
    print("\nGenerating custom visualizations...")
    visualizer.plot_mse_comparison()
    visualizer.plot_mae_comparison()
    visualizer.plot_centralized_vs_federated()
    visualizer.plot_radar_chart()
    
    print("✓ Custom visualizations completed!")


def example_interpretability_only():
    """
    Example 4: Generate only interpretability analysis
    (Feature importance and SHAP for already trained models)
    """
    # Load data and models
    print("Loading data...")
    train_data, test_data, scaler = load_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained models (example)
    print("Loading trained models...")
    lstm_model = create_model('lstm').to(device)
    # In real usage: lstm_model.load_state_dict(torch.load('lstm_model.pth'))
    
    bilstm_model = create_model('bilstm').to(device)
    # In real usage: bilstm_model.load_state_dict(torch.load('bilstm_model.pth'))
    
    cnnlstm_model = create_model('cnn-lstm').to(device)
    # In real usage: cnnlstm_model.load_state_dict(torch.load('cnnlstm_model.pth'))
    
    models_dict = {
        'LSTM': lstm_model,
        'BiLSTM': bilstm_model,
        'CNN-LSTM': cnnlstm_model
    }
    
    # Mock results (not needed for interpretability, but required for class)
    results = {
        'LSTM': {'Centralized': {'mse': 0, 'mae': 0, 'rmse': 0, 'training_time': 0}},
        'BiLSTM': {'Centralized': {'mse': 0, 'mae': 0, 'rmse': 0, 'training_time': 0}},
        'CNN-LSTM': {'Centralized': {'mse': 0, 'mae': 0, 'rmse': 0, 'training_time': 0}}
    }
    
    # Create visualizer
    visualizer = ResultsVisualizer(
        results, 
        models_dict=models_dict, 
        test_data=test_data,
        output_dir='./plots_interpretability/'
    )
    
    # Generate only interpretability plots
    print("\nGenerating interpretability analysis...")
    visualizer.plot_feature_importance()
    visualizer.plot_shap_analysis()
    
    print("✓ Interpretability analysis completed!")


def example_export_results():
    """
    Example 5: Export results to CSV for further analysis
    """
    from visualization import save_results_to_csv
    
    results = {
        'LSTM': {
            'Centralized': {'mse': 0.012, 'mae': 0.088, 'rmse': 0.111, 'training_time': 45.2},
            'Federated': {'mse': 0.015, 'mae': 0.092, 'rmse': 0.122, 'training_time': 52.1}
        },
        'BiLSTM': {
            'Centralized': {'mse': 0.011, 'mae': 0.084, 'rmse': 0.106, 'training_time': 67.5},
            'Federated': {'mse': 0.013, 'mae': 0.089, 'rmse': 0.116, 'training_time': 74.3}
        },
        'CNN-LSTM': {
            'Centralized': {'mse': 0.011, 'mae': 0.082, 'rmse': 0.105, 'training_time': 89.7},
            'Federated': {'mse': 0.013, 'mae': 0.087, 'rmse': 0.113, 'training_time': 96.5}
        }
    }
    
    print("\nExporting results to CSV...")
    df = save_results_to_csv(results, output_dir='./results/')
    
    print("\n✓ Results exported!")
    print("\nDataFrame preview:")
    print(df)
    
    # Further analysis with pandas
    print("\n--- Statistical Summary ---")
    print(df.groupby('Mode')[['MSE', 'MAE', 'RMSE']].mean())
    
    print("\n--- Best Model by MSE ---")
    best_model = df.loc[df['MSE'].idxmin()]
    print(f"Model: {best_model['Model']}, Mode: {best_model['Mode']}, MSE: {best_model['MSE']:.6f}")


def example_compare_multiple_runs():
    """
    Example 6: Compare results from multiple training runs
    """
    # Results from Run 1
    run1_results = {
        'LSTM': {
            'Centralized': {'mse': 0.012, 'mae': 0.088, 'rmse': 0.111, 'training_time': 45.2},
            'Federated': {'mse': 0.015, 'mae': 0.092, 'rmse': 0.122, 'training_time': 52.1}
        }
    }
    
    # Results from Run 2
    run2_results = {
        'LSTM': {
            'Centralized': {'mse': 0.013, 'mae': 0.089, 'rmse': 0.114, 'training_time': 44.8},
            'Federated': {'mse': 0.016, 'mae': 0.093, 'rmse': 0.126, 'training_time': 51.5}
        }
    }
    
    print("\nComparing multiple runs...")
    
    # Generate visualizations for each run
    create_visualization_report(run1_results, output_dir='./plots_run1/')
    create_visualization_report(run2_results, output_dir='./plots_run2/')
    
    # Calculate average metrics
    avg_cent_mse = (run1_results['LSTM']['Centralized']['mse'] + 
                    run2_results['LSTM']['Centralized']['mse']) / 2
    
    print(f"\n✓ Run 1 Centralized MSE: {run1_results['LSTM']['Centralized']['mse']:.6f}")
    print(f"✓ Run 2 Centralized MSE: {run2_results['LSTM']['Centralized']['mse']:.6f}")
    print(f"✓ Average Centralized MSE: {avg_cent_mse:.6f}")
    
    print("\nBoth runs visualized successfully!")


def main():
    """
    Run examples
    """
    print("="*80)
    print("VISUALIZATION MODULE - USAGE EXAMPLES")
    print("="*80)
    
    print("\n📊 Available Examples:")
    print("1. Basic usage (comparison charts only)")
    print("2. Full usage (with interpretability)")
    print("3. Custom visualization (specific charts)")
    print("4. Interpretability only (feature importance + SHAP)")
    print("5. Export results to CSV")
    print("6. Compare multiple runs")
    
    choice = input("\nEnter example number to run (1-6, or 'all'): ").strip().lower()
    
    if choice == '1':
        example_basic_usage()
    elif choice == '2':
        example_full_usage()
    elif choice == '3':
        example_custom_visualization()
    elif choice == '4':
        example_interpretability_only()
    elif choice == '5':
        example_export_results()
    elif choice == '6':
        example_compare_multiple_runs()
    elif choice == 'all':
        print("\n" + "="*80)
        print("RUNNING ALL EXAMPLES")
        print("="*80)
        example_basic_usage()
        print("\n" + "-"*80)
        example_custom_visualization()
        print("\n" + "-"*80)
        example_export_results()
        print("\n" + "-"*80)
        example_compare_multiple_runs()
        print("\n" + "="*80)
        print("✓ All examples completed!")
        print("="*80)
    else:
        print("Invalid choice. Please run again and select 1-6 or 'all'")


if __name__ == "__main__":
    # For quick testing without user input
    print("\nRunning Example 1: Basic Usage\n")
    example_basic_usage()
    
    print("\n" + "="*80)
    print("\nTo run other examples, call:")
    print("  - example_basic_usage()")
    print("  - example_full_usage()")
    print("  - example_custom_visualization()")
    print("  - example_interpretability_only()")
    print("  - example_export_results()")
    print("  - example_compare_multiple_runs()")
    print("\nOr run: python example_visualization_usage.py")
    print("="*80)