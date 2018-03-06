def anim_plot(data, lim=None, delta=1e-3):
    from jupyterthemes import jtplot
    from time import sleep
    if lim is None:
        data_min, data_max = data.min(0), data.max(0)
        lim = ((data_min[0], data_max[0]), (data_min[1], data_max[1])) 
        
    %matplotlib notebook

    plt.ion()
    fig, ax = plt.subplots()

    for i in tqdm_notebook(range(len(data))):
        ax.scatter(data[:i, 0], data[:i, 1], marker='.')
        if lim[0] is not None: ax.set_xlim(lim[0][0], lim[0][1])
        if lim[1] is not None: ax.set_ylim(lim[1][0], lim[1][1])
        fig.set_size_inches(10, 8)
        fig.canvas.draw()
        sleep(delta)
        ax.clear()

    %matplotlib inline
    jtplot.style(context='notebook', fscale=2, figsize=(15, 10))

def make_gif(line, data, interval=10, savepath=None):
    from matplotlib.animation import FuncAnimation
    
    def update(i):
        new_x, new_y = data[i]
        if new_x is not None:
            line.set_xdata(np.append(line.get_xdata(), new_x))
        if new_y is not None:
            line.set_ydata(np.append(line.get_ydata(), new_y))
    
    anim = FuncAnimation(line.figure, update, frames=np.arange(0, len(data)), interval=interval, repeat=False)
    if savepath is not None:
        anim.save(savepath, dpi=72, writer='imagemagick')
    return anim
