import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Общий стиль графиков
sns.set(style='whitegrid')
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
})

PALETTE = sns.color_palette('muted', n_colors=10)
COLORS = list(PALETTE)

METRICS_DIR = 'ablation/metrics'
OUTPUT_DIR = 'publication/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_current_figure(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')


def _style_axis(ax, xlabel: str = '', ylabel: str = '', y_locator=None, x_locator=None):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    if y_locator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))
    if x_locator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_locator))
    sns.despine(ax=ax)

def plot_ablation_learning_curves():
    print("Plotting ablation learning curves...")
    # Находим все CSV файлы
    files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
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
            linewidth = 3.0 if 'Full' in display_name else 2.0
            alpha = 1.0 if 'Full' in display_name else 0.7
            
            ax1.plot(epochs, dice_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)
            ax2.plot(epochs, err_smooth, label=display_name, color=color, linewidth=linewidth, alpha=alpha)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    ax1.set_title('Validation Dice Progression')
    _style_axis(ax1, xlabel='Epoch', ylabel='Dice', y_locator=0.02)
    ax1.legend(loc='lower right', frameon=False)

    ax2.set_title('Validation Landmark Error Progression')
    _style_axis(ax2, xlabel='Epoch', ylabel='Error (px)', y_locator=1.0)
    ax2.legend(loc='upper right', frameon=False)

    _save_current_figure('ablation_curves')
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
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models_seg))
    width = 0.35
    
    ax.bar(x - width/2, dice_scores, width, label='Dice Score', color=COLORS[0], alpha=0.9, edgecolor='black', linewidth=1.0)
    ax.bar(x + width/2, iou_scores, width, label='IoU Score', color=COLORS[1], alpha=0.9, edgecolor='black', linewidth=1.0)
    
    ax.set_title('Segmentation Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models_seg, rotation=20, ha='right')
    ax.legend(loc='lower right', frameon=False)
    ax.set_ylim(0.65, 0.95)
    _style_axis(ax, ylabel='Metric Value', y_locator=0.05)
    
    _save_current_figure('comparison_segmentation')

    # 2. Landmark Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_pts = [COLORS[7]] * (len(models_pts)-1) + [COLORS[2]]
    x_pts = np.arange(len(models_pts))

    bars = ax.bar(x_pts, pt_errors, color=colors_pts, alpha=0.9, edgecolor='black', linewidth=1.0)
    ax.set_title('Landmark Localization Accuracy (Lower is Better)')
    ax.set_xticks(x_pts)
    ax.set_xticklabels(models_pts, rotation=20, ha='right')
    _style_axis(ax, ylabel='Mean Euclidean Error (mm)', y_locator=1.0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    _save_current_figure('comparison_landmarks')

def plot_ablation_summary():
    print("Plotting ablation summary metrics...")
    files = glob.glob(os.path.join(METRICS_DIR, "*.csv"))
    results = []
    
    for file_path in sorted(files):
        fname = os.path.basename(file_path)
        name = fname.replace('ablation_', '').replace('_metrics.csv', '').replace('.csv', '')
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
    
    fig, ax1 = plt.subplots(figsize=(12, 9))
    
    y_pos = np.arange(len(res_df))
    colors_summary = [COLORS[7]] * len(res_df)
    bamnet_rows = res_df.index[res_df['Model'] == 'Full (BAMNet)'].tolist()
    for idx in bamnet_rows:
        colors_summary[list(res_df.index).index(idx)] = COLORS[2]
    
    # Горизонтальный график для Dice
    ax1.barh(y_pos, res_df['Dice'], height=0.65, color=colors_summary, alpha=0.9, edgecolor='black', linewidth=1.0, label='Dice')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(res_df['Model'], fontweight='bold')
    ax1.set_xlim(0.8, 0.9)
    ax1.set_title('Ablation Study: Summary Metrics')
    
    # Добавляем точку для ошибки на ту же ось (scale другое)
    ax2 = ax1.twiny()
    ax2.plot(res_df['Error'], y_pos, color=COLORS[3], marker='D', linestyle='', markersize=9, label='Error (px)')
    ax2.set_xlim(10, 22)
    
    _style_axis(ax1, xlabel='Dice Score')
    ax1.xaxis.set_major_locator(MultipleLocator(0.02))
    ax2.set_xlabel('Mean Landmark Error (px)', fontsize=16, color='black')
    ax2.tick_params(axis='x', labelsize=13, colors='black')
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    sns.despine(ax=ax2, left=True, bottom=True)
    
    _save_current_figure('ablation_summary')

if __name__ == "__main__":
    plot_ablation_learning_curves()
    plot_comparison_bar_charts()
    plot_ablation_summary()
    print("All plots generated successfully.")
