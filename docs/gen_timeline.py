import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

agents = [
    'Upload\nAnh',
    'Preprocessing\nAgent',
    'Classification\nAgent (ViT)',
    'Morphology\nAgent (Gemini)',
    'Retrieval\nAgent (RAG)',
    'Coordinator\nAgent',
    'Bao cao\nket qua'
]
starts = [0,   0.5, 1.7,  2.5,  2.5,  8.3, 13.1]
widths = [0.5, 1.2, 0.8,  5.8,  0.4,  4.8,  1.0]
colors = ['#264653','#2a9d8f','#e9c46a','#f4a261','#52b788','#2d6a4f','#264653']

for i, (agent, start, width, color) in enumerate(zip(agents, starts, widths, colors)):
    bar = ax.barh(0, width, left=start, height=0.55,
                  color=color, edgecolor='white', linewidth=1.5)
    cx = start + width/2
    # Short two-line label centred in bar
    ax.text(cx, 0, agent, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white',
            linespacing=1.3)
    # Time label below bar
    end = start + width
    if i > 0 and i < len(agents)-1:
        ax.text(end, -0.38, f'{end:.1f}s', ha='center', va='top',
                fontsize=7, color='#555')
        ax.plot([end, end], [-0.05, -0.28], color='#aaa', lw=0.8)

ax.set_xlim(-0.3, 15.5)
ax.set_ylim(-0.9, 0.7)
ax.set_xlabel('Thoi gian xu ly (giay)', fontsize=11)
ax.set_title('Trinh tu thuc thi cac Agent trong pipeline chan doan (tong ~14.1 giay)',
             fontsize=12, fontweight='bold', pad=12)
ax.set_yticks([])
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#ccc')

plt.tight_layout()
plt.savefig('/home/mrv24001/Leaf Detection/docs/images/agent_timeline.png',
            dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("agent_timeline.png regenerated")
