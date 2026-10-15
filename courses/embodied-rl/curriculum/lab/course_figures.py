"""Original mathematical sketches for notebooks, based on the chapter equations."""
import numpy as np
import matplotlib.pyplot as plt

def sketch(slug):
    plt.rcParams.update({'figure.facecolor':'#f7f6ef','axes.facecolor':'#f7f6ef','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(figsize=(8,3.4));green='#315d47';orange='#b36c43'
    if slug in ('probability','reinforce','sac'):
        x=np.linspace(-3,3,300)
        for mu,sig in [(0,.5),(.7,1.)]:ax.plot(x,np.exp(-.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)),label=f'mean={mu}, std={sig}')
        ax.set(xlabel='Action coordinate',ylabel='Density',title='A distribution is not one action');ax.legend()
    elif slug in ('mdp','td','actor-critic'):
        n=np.arange(12)
        for ratio in [.5,.9,.99]:ax.plot(n,ratio**n,'o-',label=f'weight ratio {ratio}')
        ax.set(xlabel='Steps into the future',ylabel='Contribution weight',title='Discounted rewards or TD-residual traces');ax.legend()
    elif slug=='ppo':
        x=np.linspace(.1,2,300)
        for adv,col in [(2,green),(-2,orange)]:
            ax.plot(x,np.minimum(x*adv,np.clip(x,.8,1.2)*adv),color=col,label=f'advantage {adv}')
        ax.axvline(.8,color='gray',ls='--');ax.axvline(1.2,color='gray',ls='--')
        ax.set(xlabel='New / old action probability ratio',ylabel='Maximized sample objective',title='PPO clips the helpful direction, not every ratio');ax.legend()
    elif slug in ('td3','ddpg','dqn','offline','grpo-flow'):
        x=np.linspace(-1,1,300);q=-(x-.3)**2
        q1=q+.2*np.sin(9*x)+.15;q2=q+.2*np.cos(9*x)+.1
        ax.plot(x,q,'--',label='Illustrative true Q');ax.plot(x,q1,alpha=.6,label='Estimate 1');ax.plot(x,q2,alpha=.6,label='Estimate 2')
        ax.plot(x,np.minimum(q1,q2),color=green,label='Minimum')
        ax.set(xlabel='Action',ylabel='Value',title='Optimizing an estimate exposes its errors');ax.legend(fontsize=8)
    elif slug in ('curricula-rewards','systems-capstone'):
        p=np.linspace(.001,.999,200)
        for target in [.3,.5,.7]:
            w=p**(4*target)*(1-p)**(4*(1-target));ax.plot(p,w/w.sum(),label=f'target {target}')
        ax.set(xlabel='Estimated reset success',ylabel='Normalized illustrative weight',title='A configuration sampler, not replay prioritization');ax.legend()
    elif slug in ('franka','agilex','r1-pro'):
        names=['arm L','arm R','torso','base'] if slug=='r1-pro' else ['arm L','arm R'] if slug=='agilex' else ['arm']
        for i,name in enumerate(names):
            ax.annotate(name,xy=(.2+.19*i,.6),xycoords='axes fraction',ha='center',bbox=dict(boxstyle='round,pad=.5',fc='#e3e9d6',ec=green))
            ax.annotate('',xy=(.5,.2),xytext=(.2+.19*i,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=green))
        ax.text(.5,.08,'One task objective · structured commands',transform=ax.transAxes,ha='center')
        ax.set_title('Concept sketch; inspect actual vendor geometry in the lab');ax.axis('off')
    else:
        positions={0:(.1,.5),1:(.4,.8),2:(.4,.2),3:(.8,.5)}
        for i,(x,y) in positions.items():
            ax.text(x,y,['state','next A','next B','future'][i],transform=ax.transAxes,ha='center',bbox=dict(boxstyle='round,pad=.6',fc='#e3e9d6',ec=green))
        for a,b in [(0,1),(0,2),(1,3),(2,3)]:ax.annotate('',xy=positions[b],xytext=positions[a],xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=green,shrinkA=25,shrinkB=30))
        ax.set_title('A one-step backup summarizes many possible futures');ax.axis('off')
    fig.tight_layout();return fig
