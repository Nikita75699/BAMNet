import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Настройки стиля
plt.style.use('bmh')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

METRICS_DIR = 'ablation/metrics'
OUTPUT_DIR = 'publication/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_ablation_learning_curves():
    print("Plotting ablation learning curves...")
    # Находим все CSV файлы
    files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Сортируем файлы для стабильности цветов
    sorted_files = sorted(files)
    
    for i, file_path in enumerate(sorted_files):
        fname = os.path.basename(file_path)
        # Очистка имени для легенды
        name = fname.replace('ablation_', '').replace('_metrics.csv', '')
        # Специальный случай для опечатки в имени файла
        if 'attentionmetrics' in name:
            name = name.replace('attentionmetrics', 'attention')
            
        name_map = {
            'full_model': 'Full Model (BAMNet)',
            'no_fusion': 'No Feature Fusion',
            'no_position_attention': 'No Position Attention',
            'no_coordinate_attention': 'No Coord Attention',
            'no_boundary_guidance': 'No Bnd Guidance',
            'no_boundary_loss': 'No Bnd Loss',
            'no_offsets': 'No Offset Refinement',
            'beta_fixed_8': 'Fixed Beta=8',
            'beta_schedule_4_8': 'Beta Schedule 4->8'
        }
        display_name = name_map.get(name, name.replace('_', ' ').capitalize())
        
        try:
            df = pd.read_csv(file_path)
            
            # Сглаживание экспоненциальным скользящим средним
            span = 5
            dice_smooth = df['val/dice'].ewm(span=span).mean()
            err_smooth = df['val_pt_err_px'].ewm(span=span).mean()
            
            epochs = df['epoch']
            
            color = COLORS[i % len(COLORS)]
            # Выделяем полную модель жирной линией
            linewidth = 2.5 if 'Full' in display_name else 1.2
            alpha = 1.0 if 'Full' in display_name else 0.7
            
            ax1.plot(epochs, dice_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)
            ax2.plot(epochs, err_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    ax1.set_title('Validation Dice Score progression', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(fontsize=8, loc='lower right', frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title('Validation Landmark Error progression', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Error (px)', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right', frameon=True, facecolor='white')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_curves.pdf'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_curves.png'), dpi=300)
    print(f"Saved to {OUTPUT_DIR}/ablation_curves.pdf")

def plot_comparison_bar_charts():
    print("Plotting comparison bar charts...")
    
    # Данные из Статьи (Таблица 2)
    models_seg = ['Swin-Unet', 'YOLO-seg (L)', 'YOLO-seg (M)', 'MaNet', 'BAMNet (Ours)']
    dice_scores = [0.8352, 0.8585, 0.8826, 0.8790, 0.8873]
    iou_scores = [0.7276, 0.7591, 0.7958, 0.8120, 0.8053]
    
    # Данные из Статьи (Таблица 3)
    models_pts = ['YOLO detect (L)', 'YOLO detect (M)', 'RT-DETR', 'YOLO-KP (L)', 'YOLO-KP (M)', 'BAMNet (Ours)']
    pt_errors = [5.32, 4.00, 5.94, 9.51, 3.69, 3.16]

    # 1. Segmentation Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_seg))
    width = 0.35
    
    ax.bar(x - width/2, dice_scores, width, label='Dice Score', color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, iou_scores, width, label='IoU Score', color='#63b5f7', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Segmentation Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_seg)
    ax.legend(loc='lower right')
    ax.set_ylim(0.65, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_segmentation.pdf'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_segmentation.png'), dpi=300)

    # 2. Landmark Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_pts = ['#95a5a6'] * (len(models_pts)-1) + ['#e74c3c'] # Highlight BAMNet
    
    bars = ax.bar(models_pts, pt_errors, color=colors_pts, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Euclidean Error (mm)', fontsize=12)
    ax.set_title('Landmark Localization Accuracy (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models_pts, rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_landmarks.pdf'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_landmarks.png'), dpi=300)

def plot_ablation_summary():
    print("Plotting ablation summary metrics...")
    files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))
    results = []
    
    for file_path in sorted(files):
        fname = os.path.basename(file_path)
        name = fname.replace('ablation_', '').replace('_metrics.csv', '')
        if 'attentionmetrics' in name:
            name = name.replace('attentionmetrics', 'attention')
            
        name_map = {
            'full_model': 'Full (BAMNet)',
            'no_fusion': 'No Fusion',
            'no_position_attention': 'No Pos Attn',
            'no_coordinate_attention': 'No Coord Attn',
            'no_boundary_guidance': 'No Bnd Guid',
            'no_boundary_loss': 'No Bnd Loss',
            'no_offsets': 'No Offsets',
            'beta_fixed_8': 'Fixed Beta=8',
            'beta_schedule_4_8': 'Beta Sched'
        }
        display_name = name_map.get(name, name.replace('_', ' ').capitalize())
        
        try:
            df = pd.read_csv(file_path)
            # Финальные значения (среднее за 5 последних эпох)
            final_dice = df['val/dice'].tail(5).mean()
            final_err = df['val_pt_err_px'].tail(5).mean()
            results.append({'Model': display_name, 'Dice': final_dice, 'Error': final_err})
        except:
            pass
    
    res_df = pd.DataFrame(results).sort_values(by='Dice', ascending=True) # Снизу вверх для красоты
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(res_df))
    
    # Горизонтальный график для Dice
    ax1.barh(y_pos, res_df['Dice'], height=0.6, color='#3498db', alpha=0.8, label='Dice')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(res_df['Model'], fontweight='bold')
    ax1.set_xlabel('Dice Score', fontsize=12, color='#3498db')
    ax1.set_xlim(0.8, 0.9)
    ax1.set_title('Ablation Study: Summary Metrics', fontsize=15, fontweight='bold')
    
    # Добавляем точку для ошибки на ту же ось (scale другое)
    ax2 = ax1.twiny()
    ax2.plot(res_df['Error'], y_pos, color='#e74c3c', marker='D', linestyle='', label='Error (px)')
    ax2.set_xlabel('Mean Landmark Error (px)', fontsize=12, color='#e74c3c')
    ax2.set_xlim(10, 22)
    
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_summary.pdf'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_summary.png'), dpi=300)

if __name__ == "__main__":
    plot_ablation_learning_curves()
    plot_comparison_bar_charts()
    plot_ablation_summary()
    print("All plots generated successfully.")
