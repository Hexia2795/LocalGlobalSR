from srpaper import runtime as rt


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.analysis.analyze import differences
    out=rt.new_run("figures")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,2,figsize=(9,7),sharex=True,sharey=True)
    for index,(seed,mode) in enumerate([(0,"Standard"),(0,"TLC"),(1,"Standard"),(1,"TLC")]):
        ax=axes.flat[index]
        for comparator,color,style,label in [("p05","#2474B5","-","Serial - P0.5"),("p1","#E58B2A","--","Serial - P1")]:
            delta=differences(seed,mode,comparator)
            ax.step(np.sort(delta),np.arange(1,len(delta)+1)/len(delta),where="post",label=label,color=color,linestyle=style)
            wins,losses=sum(delta>1e-12),sum(delta< -1e-12)
            ax.text(.04,.91 if comparator=="p05" else .72,
                    f"Mean ΔPSNR = {delta.mean():.4f} dB\nW / L / T = {wins} / {losses} / {len(delta)-wins-losses}",
                    transform=ax.transAxes,color=color,va="top",fontsize=8)
        ax.axvline(0,color="gray",linewidth=.7)
        ax.set_title(f"({chr(97+index)}) Seed {seed} / {mode}")
        ax.set_xlim(-1,1.5);ax.set_ylim(0,1);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel("Paired ΔPSNR (dB)")
    for ax in axes[:,0]:ax.set_ylabel("ECDF")
    axes[0,0].legend(loc="lower right",fontsize=8)
    fig.tight_layout();fig.savefig(out/"Fig4_ECDF.pdf");fig.savefig(out/"Fig4_ECDF.png",dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for seed,ax in enumerate(axes):
        rows=rt.read_csv(rt.ROOT/f"results/reference/conditioning_seed{seed}/conditioning_summary.csv")
        for block in ("B1","B2","B3","B4"):
            points=sorted([r for r in rows if r["block"]==block],key=lambda r:float(r["beta"]),reverse=True)
            ax.plot([float(r["beta"]) for r in points],[float(r["delta_psnr"]) for r in points],marker="o",label=block)
        ax.set_title(f"Seed {seed}");ax.set_xlabel("Global-input interpolation coefficient β");ax.invert_xaxis();ax.grid(alpha=.2)
    axes[0].set_ylabel("ΔPSNR relative to unmodified Serial (dB)")
    axes[1].legend();fig.tight_layout();fig.savefig(out/"Fig5_Conditioning.pdf");fig.savefig(out/"Fig5_Conditioning.png",dpi=300);plt.close(fig)
    print("Figures:",out)
