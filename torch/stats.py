import numpy as np
import warnings
import itertools

from matplotlib import pyplot as plt
from vai.plot import smooth_plot as _smooth_plot
from vai_.io import pickle_load, pickle_dump


def smooth_plot(y, title, together, normalize):
    y = 10 ** np.array(y)
    x = np.where(~np.isnan(y) * ~np.isinf(y))[0]

    if len(x) == 0:
        return

    if not together:
        plt.plot(x, y[x], 'b', alpha=0.3)
        _smooth_plot(x, y[x], 'b')

    else:
        _smooth_plot(x, y[x], label=title)
    if type(normalize) is tuple:
        plt.ylim(normalize)
    plt.title(title)
    plt.yscale('log')
    if not together:
        plt.show()


class ModelStats:
    def __init__(self, module, **kwargs):
        self._module = module
        self._method = kwargs.pop('method', np.median)
        self._generation = kwargs.pop('generation', 0)
        store_history = kwargs.pop('store_history', False)
        if store_history:
            warnings.warn('Storing unbounded history takes a lot of space.\n'
                          'Continue only if you are certain you have enough space.')
            self._history = {}
        else:
            self._history = None

        desc = self._module.__str__()
        self._name = kwargs.pop('name', desc[:desc.find('(')])
        parent = kwargs.pop('parent', '')
        self._full_name = parent + '/' + self._name if parent != '' else self._name

        self._children = [ModelStats(c, method=self._method, generation=self._generation + 1,
                                     name=name, parent=self._full_name)
                          for name, c in module.named_children()]
        self._fertile = (len(self._children) != 0)

        if self._fertile:
            self._grad_dict = None
        else:
            self._grad_dict = {name: None for name, _ in self._module.named_parameters()}

        self._print = lambda *args, indent=0: print('\t' * (self._generation + indent), *args)

    def read(self, dilution=0.99):
        if self._fertile:
            for c in self._children:
                c.read()
            return

        self._add(dilution)

    def show(self, display_none=True, verbose=False, cond=lambda x: x < np.inf):
        if self._fertile:
            if self._no_grads(display_none, cond):
                return

            self._print(self._name, ':')
            for c in self._children:
                c.show(display_none, verbose, cond)
            return

        self._print_result(display_none, verbose, cond)

    def reset(self):
        if self._fertile:
            for c in self._children:
                c.reset()
            return

        self._grad_dict = {name: None for name, _ in self._module.named_parameters()}

    def show_tree(self, full_names=False):
        if not full_names:
            self._print(self._name)

        if self._fertile:
            for c in self._children:
                c.show_tree(full_names)
            return

        for name, _ in self._module.named_parameters():
            if full_names:
                print('{}/{}'.format(self._full_name, name))
            else:
                self._print(name, indent=1)

    def append(self):
        for c in self._leaf_children():
            for name, g in c._grad_dict.items():
                param_name = c._full_name + '/' + name
                if param_name not in self._history.keys():
                    self._history[param_name] = []

                if g is None:
                    self._history[param_name].append(np.nan)
                else:
                    self._history[param_name].append(g[3])

    def plot_history(self, name=None, together=False, exclude_biases=True, key=None, normalize=False, last=-1):
        if normalize:
            normalize = (1e-12, 1e2)

        if name is None:
            for k, v in self._history.items():
                if exclude_biases and 'bias' in k:
                    continue
                if key is not None and key not in k:
                    continue
                if last > 0:
                    smooth_plot(v[-last:], k, together, normalize)
                else:
                    smooth_plot(v, k, together, normalize)
        else:
            if last > 0:
                smooth_plot(self._history[-last:], name, together, normalize)
            else:
                smooth_plot(self._history[name], name, together, normalize)

        if together:
                #plt.legend(loc='lower left', bbox_to_anchor=(0, -4))
                plt.show()

    def save(self, path):
        pickle_dump(path, self._history)

    def load(self, path):
        self._history = pickle_load(path, {})

    def _add(self, dilution=0.99):
        for i, (name, p) in enumerate(self._module.named_parameters()):
            if p.grad is None:
                continue

            r = self._method(p.data.cpu().numpy())
            if r != 0:
                r = np.sign(r) * np.log10(np.abs(r))

            if np.any(p.data == 0):
                continue
            v = np.abs(p.grad.data) / np.abs(p.data)
            v = self._method(v.cpu().numpy())
            if v == 0:
                continue
            v = np.log10(v)

            if self._grad_dict[name] is None:
                self._grad_dict[name] = [v, r, 1, v]
            else:
                self._grad_dict[name][0] = self._grad_dict[name][0] * dilution + v
                self._grad_dict[name][1] = self._grad_dict[name][1] * dilution + r
                self._grad_dict[name][2] = self._grad_dict[name][2] * dilution + 1
                self._grad_dict[name][3] = v

    def _print_result(self, display_none, verbose, cond):
        if self._no_grads(display_none, cond):
            return

        _print = self._print
        if verbose:
            _print(self._name, '-', self._module, ':')
        else:
            _print(self._name, ':')

        for i, (name, _) in enumerate(self._module.named_parameters()):
            if self._grad_dict[name] is None:
                if not display_none:
                    continue
                v = r = 'None'
            else:
                v = self._grad_dict[name][0] / self._grad_dict[name][2]
                r = self._grad_dict[name][1] / self._grad_dict[name][2]
            if self._grad_dict[name] is None:
                _print(name, '---> None', indent=1)
            else:
                if verbose and cond(v):
                    _print(name, '--> {:.2f}, ({:.2f})'.format(v, r), indent=1)
                elif cond(v):
                    _print(name, '--> {:.2f}'.format(v), indent=1)

    def _no_grads(self, display_none, cond):
        if self._fertile:
            return all(c._no_grads(display_none, cond) for c in self._children)
        elif display_none:
            return all(g is None or ~cond(g[0] / g[2]) for g in self._grad_dict.values())
        else:
            return all(~cond(g[0] / g[2]) for g in self._grad_dict.values() if g is not None)

    def _leaf_children(self):
        if self._fertile:
            return list(itertools.chain(*[c._leaf_children() for c in self._children]))

        return [self]

    def __call__(self, name):
        new_names = name.split('/')

        if self._fertile:
            name = new_names[0]
            new_names = '/'.join(new_names[1:])

            children_found = [c for c in self._children if c._name == name]

            if new_names == '':
                return children_found[0] if len(children_found) > 0 else None

            children_found = [c(new_names) for c in children_found]
            children_found = [c for c in children_found if c is not None]

            return children_found[0] if len(children_found) > 0 else None
