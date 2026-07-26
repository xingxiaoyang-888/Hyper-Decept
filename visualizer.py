import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None
    print("pip install shap")

class CognitiveVisualizer:
    def __init__(self, X_super, y_true, y_role, feature_names, save_dir=r"E:\fraud-detection2\Multi-agent-fraud-game-detection\new_result"):
       
        self.X = X_super
        self.y = y_true
        self.y_role = y_role  
        self.feature_names = feature_names
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
                # Convert matrix to DataFrame for easier slicing
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        
        # Use tactical role labels instead of binary labels
        self.df['Tactical_Role'] = self.y_role
        
        self.psycho_feats = [
            'Empathy_Gap_Mean', 'Empathy_Gap_Max',
            'Dark_Triad_Mean', 'Dark_Triad_Max',
            'Contagion_Mean', 'Contagion_Max',
            'Volatility_Mean', 'Volatility_Max'
        ]

        
        self.role_palette = {}
        unique_roles = sorted(set(self.y_role)) if self.y_role is not None else []
        colors = sns.color_palette("husl", max(len(unique_roles), 4))
        for i, r in enumerate(unique_roles):
            self.role_palette[r] = colors[i]

    def plot_radar_chart(self):
        missing = [f for f in self.psycho_feats if f not in self.df.columns]
        if missing:
            return None

        angles = np.linspace(0, 2 * np.pi, len(self.psycho_feats), endpoint=False).tolist()
        angles += angles[:1] 
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for role, color in self.role_palette.items():
            role_df = self.df[self.df['Tactical_Role'] == role]
            if len(role_df) == 0:
                continue
                
            mean_vals = role_df[self.psycho_feats].mean().values
            mean_vals = np.concatenate((mean_vals, [mean_vals[0]]))
            
            
            line_style = 'solid' if 'Leader' in role else 'dashed'
            alpha_val = 0.3 if 'Leader' in role else 0.15
            
            ax.plot(angles, mean_vals, color=color, linewidth=2.5, linestyle=line_style, label=role)
            ax.fill(angles, mean_vals, color=color, alpha=alpha_val)
        
        ax.set_xticks(angles[:-1])
        short_names = ['Empathy\nMean', 'Empathy\nMax', 'Dark\nTriad\nMean', 'Dark\nTriad\nMax',
                        'Contagion\nMean', 'Contagion\nMax', 'Volatility\nMean', 'Volatility\nMax']
        ax.set_xticklabels(short_names, fontsize=12, fontweight='bold')
        
        plt.title('Psychological Profiling Radar by Tactical Role\n(Digital Fingerprints of Deception)', size=15, y=1.1, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "psycho_radar_chart_tactical.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_kde_distribution(self, target_feature='Empathy_Gap_Mean'):
        
        print(f"\n {target_feature} ...")
        if target_feature not in self.df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=self.df, x=target_feature, hue='Tactical_Role', 
            fill=True, common_norm=False, palette=self.role_palette, 
            alpha=0.4, linewidth=2, ax=ax
        )
        
        plt.title(f'KDE Distribution of {target_feature} by Tactical Role\n(Deconstructing the Camouflage)', fontsize=14, fontweight='bold')
        plt.xlabel(target_feature, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.35, 1))
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"kde_tactical_{target_feature}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_3d_scatter(self):
       
        missing = [f for f in self.psycho_feats if f not in self.df.columns]
        if missing:
            return None
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        self.df['_label'] = ['Bot' if v == 1 else 'Human' for v in self.y]
        good_df = self.df[self.df['_label'] == 'Human']
        bad_df = self.df[self.df['_label'] == 'Bot']
        ax.scatter(good_df[self.psycho_feats[0]], good_df[self.psycho_feats[1]], good_df[self.psycho_feats[2]],
                   c='#3498db', label='Human', marker='o', s=50, alpha=0.6)
        ax.scatter(bad_df[self.psycho_feats[0]], bad_df[self.psycho_feats[1]], bad_df[self.psycho_feats[2]],
                   c='#e74c3c', label='Bot', marker='X', s=80, alpha=0.9)

        ax.set_xlabel(self.psycho_feats[0], fontweight='bold')
        ax.set_ylabel(self.psycho_feats[1], fontweight='bold')
        ax.set_zlabel(self.psycho_feats[2], fontweight='bold')
        plt.title('3D Psychological Isolation Space', fontsize=16)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "psycho_3d_scatter.png")
        plt.savefig(save_path, dpi=300)
        return fig

    def plot_shap_summary(self, trained_xgb_model):
        if shap is None:
            return None
            
       
        X_df = self.df.drop(columns=['Tactical_Role', '_label'], errors='ignore')
        
        
        explainer = shap.TreeExplainer(trained_xgb_model)
        shap_values = explainer.shap_values(X_df)
        
        fig = plt.figure(figsize=(12, 8))
       
        shap.summary_plot(shap_values, X_df, plot_type="dot", max_display=15, show=False)
        
        plt.title('SHAP Feature Attribution (Impact on Detection)', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "shap_summary_plot.png")
        plt.savefig(save_path, dpi=300)
        return fig

    def generate_all_reports(self, trained_xgb_model=None):
        
        print("\n" + "="*50)
        print(" visualizer")
        print("="*50)
        
        self.plot_radar_chart()
        self.plot_kde_distribution('Empathy_Gap_Mean')
        self.plot_kde_distribution('Dark_Triad_Mean')
        self.plot_kde_distribution('Contagion_Mean')
        self.plot_3d_scatter()
        
        if trained_xgb_model is not None:
            self.plot_shap_summary(trained_xgb_model)
            
        print(f"\n save_path: {self.save_dir}")