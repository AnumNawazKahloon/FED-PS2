import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
from sklearn.inspection import permutation_importance
import seaborn as sns
from config import DATA
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ResultsVisualizer:
    """
    Comprehensive visualization for model comparison and interpretability
    """
    def __init__(self, results, models_dict=None, test_data=None, output_dir='/Users/virk/Parma/FED-PS2/plots/'):
        """
        Args:
            results: Dictionary with model results
            models_dict: Dictionary of trained models for interpretability
            test_data: Test dataset for SHAP analysis
            output_dir: Directory to save plots
        """
        self.results = results
        self.models_dict = models_dict or {}
        self.test_data = test_data
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes
        self.centralized_color = '#2E86AB'
        self.federated_color = '#A23B72'
        self.model_colors = {
            'LSTM': '#E63946',
            'BiLSTM': '#F77F00',
            'CNN-LSTM': '#06A77D'
        }
    
    def plot_all(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # 1. Comparison charts
        self.plot_mse_comparison()
        self.plot_mae_comparison()
        self.plot_rmse_comparison()
        self.plot_training_time_comparison()
        self.plot_accuracy_bars()
        
        # 2. Centralized vs Federated
        self.plot_centralized_vs_federated()
        self.plot_performance_degradation()
        
        # 3. Model comparison
        self.plot_model_ranking()
        self.plot_radar_chart()
        
        # 4. Training curves (if available)
        self.plot_combined_metrics()
        
        # 5. Feature importance and SHAP
        if self.models_dict and self.test_data:
            self.plot_feature_importance()
            self.plot_shap_analysis()
        
        print(f"\n✓ All plots saved to: {self.output_dir}")
        print("="*60)
    
    def plot_mse_comparison(self):
        """Plot MSE comparison across all models"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        centralized_mse = [self.results[m]['Centralized']['mse'] for m in models]
        federated_mse = [self.results[m]['Federated']['mse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, centralized_mse, width, 
                      label='Centralized', color=self.centralized_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, federated_mse, width, 
                      label='Federated', color=self.federated_color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
        ax.set_title('MSE Comparison: Centralized vs Federated Learning', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}mse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ MSE comparison chart saved")
    
    def plot_mae_comparison(self):
        """Plot MAE comparison across all models"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        centralized_mae = [self.results[m]['Centralized']['mae'] for m in models]
        federated_mae = [self.results[m]['Federated']['mae'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, centralized_mae, width, 
                      label='Centralized', color=self.centralized_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, federated_mae, width, 
                      label='Federated', color=self.federated_color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
        ax.set_title('MAE Comparison: Centralized vs Federated Learning', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}mae_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ MAE comparison chart saved")
    
    def plot_rmse_comparison(self):
        """Plot RMSE comparison across all models"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        centralized_rmse = [self.results[m]['Centralized']['rmse'] for m in models]
        federated_rmse = [self.results[m]['Federated']['rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, centralized_rmse, width, 
                      label='Centralized', color=self.centralized_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, federated_rmse, width, 
                      label='Federated', color=self.federated_color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=12, fontweight='bold')
        ax.set_title('RMSE Comparison: Centralized vs Federated Learning', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}rmse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ RMSE comparison chart saved")
    
    def plot_training_time_comparison(self):
        """Plot training time comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        centralized_time = [self.results[m]['Centralized']['training_time'] for m in models]
        federated_time = [self.results[m]['Federated']['training_time'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, centralized_time, width, 
                      label='Centralized', color=self.centralized_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, federated_time, width, 
                      label='Federated', color=self.federated_color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Comparison: Centralized vs Federated Learning', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}training_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training time comparison chart saved")
    
    def plot_accuracy_bars(self):
        """Plot accuracy metrics in grouped bars"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(self.results.keys())
        
        # Centralized
        cent_mse = [self.results[m]['Centralized']['mse'] for m in models]
        cent_mae = [self.results[m]['Centralized']['mae'] for m in models]
        cent_rmse = [self.results[m]['Centralized']['rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0].bar(x - width, cent_mse, width, label='MSE', color='#E63946', alpha=0.8)
        axes[0].bar(x, cent_mae, width, label='MAE', color='#F77F00', alpha=0.8)
        axes[0].bar(x + width, cent_rmse, width, label='RMSE', color='#06A77D', alpha=0.8)
        axes[0].set_xlabel('Model', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Error Metric', fontsize=11, fontweight='bold')
        axes[0].set_title('Centralized Learning - All Metrics', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Federated
        fed_mse = [self.results[m]['Federated']['mse'] for m in models]
        fed_mae = [self.results[m]['Federated']['mae'] for m in models]
        fed_rmse = [self.results[m]['Federated']['rmse'] for m in models]
        
        axes[1].bar(x - width, fed_mse, width, label='MSE', color='#E63946', alpha=0.8)
        axes[1].bar(x, fed_mae, width, label='MAE', color='#F77F00', alpha=0.8)
        axes[1].bar(x + width, fed_rmse, width, label='RMSE', color='#06A77D', alpha=0.8)
        axes[1].set_xlabel('Model', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Error Metric', fontsize=11, fontweight='bold')
        axes[1].set_title('Federated Learning - All Metrics', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}accuracy_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Accuracy bars chart saved")
    
    def plot_centralized_vs_federated(self):
        """Plot centralized vs federated for each model"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Centralized vs Federated Learning Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        models = list(self.results.keys())
        metrics = ['mse', 'mae', 'rmse', 'training_time']
        titles = ['MSE', 'MAE', 'RMSE', 'Training Time (s)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            centralized = [self.results[m]['Centralized'][metric] for m in models]
            federated = [self.results[m]['Federated'][metric] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, centralized, width, label='Centralized', 
                  color=self.centralized_color, alpha=0.8)
            ax.bar(x + width/2, federated, width, label='Federated', 
                  color=self.federated_color, alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}centralized_vs_federated.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Centralized vs Federated comparison chart saved")
    
    def plot_performance_degradation(self):
        """Plot performance degradation from centralized to federated"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        degradations = []
        
        for model in models:
            cent_mse = self.results[model]['Centralized']['mse']
            fed_mse = self.results[model]['Federated']['mse']
            degradation = ((fed_mse - cent_mse) / cent_mse) * 100
            degradations.append(degradation)
        
        colors = [self.model_colors.get(m, '#888888') for m in models]
        bars = ax.bar(models, degradations, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
        ax.set_title('MSE Degradation: Centralized → Federated', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}%',
                   ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}performance_degradation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Performance degradation chart saved")
    
    def plot_model_ranking(self):
        """Plot model rankings across different metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Rankings by Metric', fontsize=16, fontweight='bold', y=0.995)
        
        metrics = ['mse', 'mae', 'rmse', 'training_time']
        titles = ['MSE (Lower is Better)', 'MAE (Lower is Better)', 
                 'RMSE (Lower is Better)', 'Training Time (Faster is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Get all results for this metric
            all_results = []
            for model in self.results.keys():
                for mode in ['Centralized', 'Federated']:
                    all_results.append({
                        'label': f"{model}\n{mode}",
                        'value': self.results[model][mode][metric],
                        'model': model
                    })
            
            # Sort by value
            all_results = sorted(all_results, key=lambda x: x['value'])
            
            labels = [r['label'] for r in all_results]
            values = [r['value'] for r in all_results]
            colors_list = [self.model_colors.get(r['model'], '#888888') for r in all_results]
            
            bars = ax.barh(labels, values, color=colors_list, alpha=0.7, edgecolor='black')
            ax.set_xlabel(title.split('(')[0], fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.4f}' if metric != 'training_time' else f'{width:.1f}s',
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}model_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Model ranking chart saved")
    
    def plot_radar_chart(self):
        """Plot radar chart for model comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        fig.suptitle('Model Performance Radar Chart', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        
        # Normalize metrics (inverse for errors, keep time as is)
        for mode_idx, mode in enumerate(['Centralized', 'Federated']):
            ax = axes[mode_idx]
            
            categories = ['MSE', 'MAE', 'RMSE', 'Speed']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_title(f'{mode} Learning', fontsize=13, fontweight='bold', pad=20)
            ax.grid(True)
            
            for model in models:
                # Get metrics (normalize to 0-1, inverse for errors)
                mse = self.results[model][mode]['mse']
                mae = self.results[model][mode]['mae']
                rmse = self.results[model][mode]['rmse']
                time = self.results[model][mode]['training_time']
                
                # Normalize (inverse for errors - lower is better)
                max_mse = max([self.results[m][mode]['mse'] for m in models])
                max_mae = max([self.results[m][mode]['mae'] for m in models])
                max_rmse = max([self.results[m][mode]['rmse'] for m in models])
                max_time = max([self.results[m][mode]['training_time'] for m in models])
                
                values = [
                    1 - (mse / max_mse),
                    1 - (mae / max_mae),
                    1 - (rmse / max_rmse),
                    1 - (time / max_time)
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model, color=self.model_colors.get(model, '#888888'))
                ax.fill(angles, values, alpha=0.15, 
                       color=self.model_colors.get(model, '#888888'))
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Radar chart saved")
    
    def plot_combined_metrics(self):
        """Plot combined metrics comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.15
        
        metrics_data = {
            'Cent MSE': ([self.results[m]['Centralized']['mse'] for m in models], -2*width, '#E63946'),
            'Fed MSE': ([self.results[m]['Federated']['mse'] for m in models], -width, '#F77F00'),
            'Cent MAE': ([self.results[m]['Centralized']['mae'] for m in models], 0, '#06A77D'),
            'Fed MAE': ([self.results[m]['Federated']['mae'] for m in models], width, '#4CC9F0'),
            'Cent RMSE': ([self.results[m]['Centralized']['rmse'] for m in models], 2*width, '#7209B7'),
        }
        
        for label, (data, offset, color) in metrics_data.items():
            ax.bar(x + offset, data, width, label=label, color=color, alpha=0.7)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Metric Value', fontsize=12, fontweight='bold')
        ax.set_title('Combined Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(ncol=3, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}combined_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Combined metrics chart saved")
    
    def plot_feature_importance(self):
        """Plot feature importance for each model using permutation importance"""
        print("\n" + "-"*60)
        print("Computing Feature Importance...")
        print("-"*60)
        
        if not self.models_dict or not self.test_data:
            print("⚠ Models or test data not available for feature importance")
            return
        
        feature_names = DATA['features']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract test data
        X_test = self.test_data.tensors[0].numpy()
        y_test = self.test_data.tensors[1].numpy()
        
        fig, axes = plt.subplots(len(self.models_dict), 1, 
                                figsize=(12, 5*len(self.models_dict)))
        if len(self.models_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models_dict.items()):
            ax = axes[idx]
            
            # Compute permutation importance
            model.eval()
            
            # Create a wrapper function for sklearn's permutation_importance
            def model_predict(X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    predictions = model(X_tensor).cpu().numpy().flatten()
                return predictions
            
            # Compute importance
            result = permutation_importance(
                model_predict, X_test, y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Sort by importance
            sorted_idx = result.importances_mean.argsort()[::-1]
            
            # Plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            ax.barh(range(len(feature_names)), 
                   result.importances_mean[sorted_idx],
                   xerr=result.importances_std[sorted_idx],
                   color=colors[sorted_idx], alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
            ax.set_xlabel('Permutation Importance', fontsize=11, fontweight='bold')
            ax.set_title(f'Feature Importance - {model_name}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3)
            
            print(f"✓ Feature importance computed for {model_name}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance chart saved")
    
    def plot_shap_analysis(self):
        """Plot SHAP values for model interpretability"""
        print("\n" + "-"*60)
        print("Computing SHAP Values...")
        print("-"*60)
        
        if not self.models_dict or not self.test_data:
            print("⚠ Models or test data not available for SHAP analysis")
            return
        
        feature_names = DATA['features']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract test data (use subset for SHAP - it's computationally expensive)
        X_test = self.test_data.tensors[0][:100].numpy()  # Use first 100 samples
        
        for model_name, model in self.models_dict.items():
            print(f"Computing SHAP for {model_name}...")
            
            model.eval()
            
            # Create a wrapper function for SHAP
            def model_predict(X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    predictions = model(X_tensor).cpu().numpy().flatten()
                return predictions
            
            try:
                # Create SHAP explainer (using KernelExplainer for neural networks)
                # Use a subset of data as background
                background = X_test[:50]
                explainer = shap.KernelExplainer(model_predict, background)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_test[:20])  # Explain 20 samples
                
                # Summary plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 12))
                
                # SHAP Summary Plot (Bar)
                plt.sca(axes[0])
                shap.summary_plot(shap_values, X_test[:20], 
                                feature_names=feature_names,
                                plot_type="bar", show=False)
                axes[0].set_title(f'SHAP Feature Importance - {model_name}', 
                                fontsize=12, fontweight='bold', pad=15)
                
                # SHAP Summary Plot (Beeswarm)
                plt.sca(axes[1])
                shap.summary_plot(shap_values, X_test[:20], 
                                feature_names=feature_names,
                                show=False)
                axes[1].set_title(f'SHAP Value Distribution - {model_name}', 
                                fontsize=12, fontweight='bold', pad=15)
                
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}shap_{model_name.lower().replace("-", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ SHAP analysis completed for {model_name}")
                
            except Exception as e:
                print(f"⚠ SHAP analysis failed for {model_name}: {str(e)}")
                print(f"  Continuing with other models...")
        
        print("✓ SHAP analysis charts saved")


def save_results_to_csv(results, output_dir='/Users/virk/Parma/FED-PS1/plots/'):
    """Save results to CSV file for further analysis"""
    import pandas as pd
    
    data = []
    for model_name, modes in results.items():
        for mode, metrics in modes.items():
            data.append({
                'Model': model_name,
                'Mode': mode,
                'MSE': metrics['mse'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'Training_Time': metrics['training_time']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}results_summary.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir}results_summary.csv")
    
    return df


def create_visualization_report(results, models_dict=None, test_data=None, output_dir='/Users/virk/Parma/FED-PS1/plots/'):
    """
    Main function to create all visualizations
    
    Args:
        results: Dictionary with model results from main.py
        models_dict: Dictionary of trained models (optional, for interpretability)
        test_data: Test dataset (optional, for interpretability)
        output_dir: Directory to save plots
    
    Example:
        from visualization import create_visualization_report
        
        # After training in main.py
        models_dict = {
            'LSTM': lstm_centralized_model,
            'BiLSTM': bilstm_centralized_model,
            'CNN-LSTM': cnnlstm_centralized_model
        }
        
        create_visualization_report(results, models_dict, test_data)
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION REPORT")
    print("="*80)
    
    # Create visualizer
    visualizer = ResultsVisualizer(results, models_dict, test_data, output_dir)
    
    # Generate all plots
    visualizer.plot_all()
    
    # Save results to CSV
    df = save_results_to_csv(results, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION REPORT COMPLETED")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. mse_comparison.png - MSE comparison chart")
    print("  2. mae_comparison.png - MAE comparison chart")
    print("  3. rmse_comparison.png - RMSE comparison chart")
    print("  4. training_time_comparison.png - Training time comparison")
    print("  5. accuracy_bars.png - Accuracy metrics grouped bars")
    print("  6. centralized_vs_federated.png - Mode comparison")
    print("  7. performance_degradation.png - Degradation analysis")
    print("  8. model_ranking.png - Rankings by metric")
    print("  9. radar_chart.png - Radar comparison")
    print(" 10. combined_metrics.png - All metrics combined")
    print(" 11. feature_importance.png - Feature importance (if models provided)")
    print(" 12. shap_*.png - SHAP analysis (if models provided)")
    print(" 13. results_summary.csv - Results in CSV format")
    print("="*80 + "\n")
    
    return df