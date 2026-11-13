"""Train independent spatial experts together with the original distance loss.

Models have separate parameters and Adam moments. Sum per-model losses so
batch size does not rescale gradients. CUDA batched matrix multiplication may
round differently from separate training. Exports retain every model and seed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from kilix_bench.braille_layouts import LAYOUTS, REGIONS, WIDTHS, crop_field


def position(torch, xy):
    values = [xy[..., :1] * 2 - 1, xy[..., 1:] - 1]
    for k in range(6):
        phase = 2 * math.pi * 2**k * xy
        values += [torch.sin(phase), torch.cos(phase)]
    return torch.cat(values, -1)


def code(torch, scalar, masks):
    bits = ((scalar[..., None] >> torch.arange(21, device=scalar.device)) & 1).float()
    return 2 * torch.cat([bits, torch.remainder(bits @ masks.T, 2)], -1) - 1


def field(torch, weights, biases, features, xy):
    count, glyphs, points, _ = xy.shape
    value = torch.cat([position(torch, xy), features[:, :, None, :].expand(-1, -1, points, -1)], -1)
    value = value.reshape(count, glyphs * points, 90)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        value = torch.bmm(value, weight.transpose(1, 2)) + bias[:, None, :]
        if i < 3:
            value = torch.relu(value)
    return value.reshape(count, glyphs, points)


def losses(torch, weights, biases, labelled, xy, wanted, unlabelled, pu, truth):
    fitted = field(torch, weights, biases, labelled, xy)
    loss = (fitted - wanted).abs().mean((1, 2))
    su = torch.sigmoid(8 * field(torch, weights, biases, unlabelled, pu))
    mean = su.mean(2)
    dd = (su[:, :, None] - su[:, None]).abs().mean(-1)
    dt = (su[:, :, None] - truth[:, None]).abs().mean(-1)
    u = su.shape[1]
    # Diagonal entries are zero distance and contribute exactly 0.12 each.
    distinct = (torch.relu(.12 - dd).sum((1, 2)) - .12*u) / (u*(u-1))
    return loss + (torch.relu(.08-mean) + torch.relu(mean-.25)).mean(1) + distinct + torch.relu(.12-dt).mean((1, 2))


def verify(torch):
    """Independent Sequential forward and original boolean-mask regularizer."""
    torch.manual_seed(731)
    for width, glyphs in ((1, 2), (2, 4), (4, 2), (6, 4)):
        nets = [torch.nn.Sequential(torch.nn.Linear(90, width), torch.nn.ReLU(),
                torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                torch.nn.ReLU(), torch.nn.Linear(width, 1)) for _ in range(2)]
        layers = [[layer for layer in net if isinstance(layer, torch.nn.Linear)] for net in nets]
        weights = [torch.nn.Parameter(torch.stack([ls[i].weight.detach() for ls in layers])) for i in range(4)]
        biases = [torch.nn.Parameter(torch.stack([ls[i].bias.detach() for ls in layers])) for i in range(4)]
        labelled, unlabelled = torch.randn(2, glyphs, 64), torch.randn(2, 64, 64)
        xy, pu = torch.rand(2, glyphs, 256, 2), torch.rand(2, 64, 256, 2)
        wanted, truth = torch.randn(2, glyphs, 256), torch.randint(0, 2, (2, glyphs, 256)).float()
        batch_opt = torch.optim.Adam(weights+biases, lr=.002)
        opts = [torch.optim.Adam(net.parameters(), lr=.002) for net in nets]
        for step in range(3):
            batch_opt.zero_grad()
            values = losses(torch, weights, biases, labelled, xy, wanted, unlabelled, pu, truth)
            values.sum().backward()
            for m, net in enumerate(nets):
                opts[m].zero_grad()
                def separate(cf, points):
                    return net(torch.cat([position(torch, points), cf[:, None].expand(-1, points.shape[1], -1)], -1)).squeeze(-1)
                su = torch.sigmoid(8*separate(unlabelled[m], pu[m]))
                mean = su.mean(1)
                dd = (su[:, None]-su[None]).abs().mean(-1)
                dt = (su[:, None]-truth[m][None]).abs().mean(-1)
                loss = (separate(labelled[m], xy[m])-wanted[m]).abs().mean()
                loss += (torch.relu(.08-mean)+torch.relu(mean-.25)).mean()
                loss += torch.relu(.12-dd)[~torch.eye(64, dtype=torch.bool)].mean() + torch.relu(.12-dt).mean()
                torch.testing.assert_close(values[m], loss, atol=2e-6, rtol=2e-5)
                loss.backward()
                for i, layer in enumerate(layers[m]):
                    torch.testing.assert_close(weights[i].grad[m], layer.weight.grad, atol=2e-6, rtol=2e-4)
                    torch.testing.assert_close(biases[i].grad[m], layer.bias.grad, atol=2e-6, rtol=2e-4)
                opts[m].step()
            batch_opt.step()
            for m in range(2):
                for i, layer in enumerate(layers[m]):
                    torch.testing.assert_close(weights[i][m], layer.weight, atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(biases[i][m], layer.bias, atol=3e-6, rtol=3e-5)
    print('independent loss, gradients and three Adam updates: passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--neuronte-home', type=Path, required=True)
    parser.add_argument('--layout', choices=LAYOUTS, default='dots')
    parser.add_argument('--patterns', type=int, choices=(2, 4), default=2)
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        verify(torch)
        return
    if args.steps < 1:
        parser.error('steps must be positive')
    torch.set_default_device(args.device)
    home = args.neuronte_home.resolve()
    sys.path.insert(0, str(home / 'train'))
    import glyphs
    import strokes
    seeds = [int(s) for s in args.seeds.split(',')]
    parts = [part for part in LAYOUTS[args.layout] if 1 << REGIONS[part][0].bit_count() == args.patterns]
    members = [(part, seed) for seed in seeds for part in parts]
    if not members or len(set(seeds)) != len(seeds):
        parser.error('empty group or duplicate seeds')
    args.directory.mkdir(parents=True, exist_ok=True)
    width, count, g, n = WIDTHS[args.layout], len(members), args.patterns, 256
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {p: sha(home/p) for p in glyphs.sources('braille') + ['train/train.py', 'train/checkpoint.py', 'font/code-masks.txt']}
    identity = {'schema': 'kilix-bench.braille-batch/v1', 'members': members, 'width': width,
                'steps': args.steps, 'sources': sources, 'trainer_sha256': sha(Path(__file__)),
                'layouts_sha256': sha(Path(__file__).with_name('braille_layouts.py')),
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'compiled': args.compile,
                'device': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else args.device,
                'tf32': False, 'loss_reduction': 'sum of independent per-model losses'}
    print(json.dumps(identity), flush=True)
    teacher = {ord(c): shapes for c, _, shapes in glyphs.charset('braille')}
    labels, fields = [], []
    for part, seed in members:
        mask = REGIONS[part][0]
        lab = [0x2800 | bits for bits in range(256) if not bits & ~mask]
        labels.append(lab)
        fields.append(np.stack([crop_field(glyphs.field(teacher[c], n, glyphs.PEN, .15).astype(np.float16).astype(np.float32), REGIONS[part]) for c in lab]).astype(np.float16))
    target_field = torch.tensor(np.stack(fields)).reshape(-1)
    lab = torch.tensor(labels)
    offsets = torch.arange(count*g).reshape(count, g, 1) * (n+1)*(2*n+1)
    def target(xy):
        u, v = xy[..., 0]*n, xy[..., 1]*n
        i, j = u.floor().clamp(0, n-1), v.floor().clamp(0, 2*n-1)
        fu, fv = u-i, v-j
        k = offsets + j.long()*(n+1) + i.long()
        a, b, c, d = (target_field[k+offset].float() for offset in (0, 1, n+1, n+2))
        return (a*(1-fu)+b*fu)*(1-fv)+(c*(1-fu)+d*fu)*fv
    masks = strokes.masks()
    mb = torch.tensor([[(m>>i)&1 for i in range(21)] for m in masks], dtype=torch.float32)
    cf = code(torch, lab, mb)
    nets, generators = {}, {}
    for seed in seeds:
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(90, width), torch.nn.ReLU(),
                torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                torch.nn.ReLU(), torch.nn.Linear(width, 1))
        nets[seed] = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
        gen = torch.Generator(device=args.device)
        gen.set_state(torch.cuda.get_rng_state() if args.device.startswith('cuda') else torch.get_rng_state())
        generators[seed] = gen
    weights = [torch.nn.Parameter(torch.stack([nets[seed][i].weight.detach() for _, seed in members])) for i in range(4)]
    biases = [torch.nn.Parameter(torch.stack([nets[seed][i].bias.detach() for _, seed in members])) for i in range(4)]
    opt = torch.optim.Adam(weights+biases, lr=.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    first = 0
    ckpt = args.directory / f'{args.layout}-g{g}.pt'
    if args.resume:
        saved = torch.load(ckpt, weights_only=False, map_location=args.device)
        if saved['identity'] != identity:
            raise ValueError('checkpoint identity differs')
        with torch.no_grad():
            for p, q in zip(weights+biases, saved['parameters']):
                p.copy_(q)
        opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler'])
        for seed, state in saved['rng'].items():
            generators[seed].set_state(state.cpu())
        first = saved['step']
        values = saved['losses']
    loss_fn = lambda *values: losses(torch, *values)
    if args.compile:
        loss_fn = torch.compile(loss_fn, fullgraph=True)
    span = torch.tensor([1., 2.])
    started = time.monotonic()
    for step in range(first, args.steps):
        draws = {}
        for seed in seeds:
            gen = generators[seed]
            draws[seed] = (torch.rand(g, 1024, 2, generator=gen)*span,
                           torch.randint(0, 0x110000, (64,), generator=gen),
                           torch.rand(1, 256, 2, generator=gen)*span)
        cand = torch.stack([draws[s][0] for _, s in members])
        v = torch.stack([draws[s][1] for _, s in members])
        pu = torch.stack([draws[s][2] for _, s in members])
        v = torch.where((v[..., None] == lab[:, None, :]).any(-1) | ((v >= 0xd800) & (v <= 0xdfff)), v+0x1000, v)
        tc = target(cand)
        near = tc.abs().topk(128, dim=2, largest=False).indices
        xy = torch.cat([cand[:, :, :128], torch.gather(cand, 2, near[..., None].expand(-1, -1, -1, 2))], 2)
        wanted = torch.cat([tc[:, :, :128], torch.gather(tc, 2, near)], 2)
        truth = (target(pu.expand(-1, g, -1, -1)) > 0).float()
        opt.zero_grad()
        loss = loss_fn(weights, biases, cf, xy, wanted, code(torch, v, mb), pu.expand(-1, 64, -1, -1), truth)
        loss.sum().backward()
        opt.step()
        sched.step()
        if (step+1) % 1000 == 0 or step+1 == args.steps:
            values = loss.detach().cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite training loss')
            print(json.dumps({'step': step+1, 'seconds': time.monotonic()-started, 'losses': values}), flush=True)
        if (step+1) % 2000 == 0 or step+1 == args.steps:
            temporary = ckpt.with_suffix('.tmp')
            torch.save({'identity': identity, 'step': step+1, 'parameters': [p.detach() for p in weights+biases],
                        'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                        'losses': values,
                        'rng': {s: gen.get_state() for s, gen in generators.items()}}, temporary)
            temporary.replace(ckpt)
    for m, (part, seed) in enumerate(members):
        mask, x0, x1, y0, y1 = REGIONS[part]
        experiment = dict(identity, part=part, seed=seed, hidden_width=width, x_domain=[x0, x1], y_domain=[y0, y1],
                          file_tau=.15/(x1-x0), original_target_tau=.15, training_steps=args.steps,
                          input_mapping=f'U+2800 | ((scalar-U+2800) & 0x{mask:02x})', final_loss=values[m])
        doc = {'format': 'neuronte-weights', 'version': 1,
               'shape': {'position_octaves': 6, 'code_bits': 64, 'hidden_width': width, 'hidden_layers': 3,
                         'tau': .15/(x1-x0), 'parity_masks': [f'0x{x:06X}' for x in masks]},
               'training': {'set': 'braille-'+part, 'labelled': ' '.join(f'U+{x:04X}' for x in labels[m]),
                            'sources_sha256': sources, 'pen_radius': glyphs.PEN, 'grid': n, 'steps': args.steps,
                            'seed': seed, 'unlabelled_weight': 1., 'experiment': experiment},
               'layers': {name: {'rows': w.shape[1], 'cols': w.shape[2], 'weight': w[m].detach().cpu().flatten().tolist(),
                                 'bias': b[m].detach().cpu().tolist()}
                          for name, w, b in zip(('l1', 'l2', 'l3', 'out'), weights, biases)}}
        (args.directory / f'{part}-s{seed}.json').write_text(json.dumps(doc)+'\n')


if __name__ == '__main__':
    main()
