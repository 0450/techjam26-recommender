"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个官方特征域。扩展域是 item-side / 序列 / 时间，不是已被证伪的静态用户分桶。
BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
EXTRA_FIELDS = ['hour_bucket', 'pop_bucket', 'last_video', 'last_pos_video']
FIELDS = BASE_FIELDS  # official 5-field default; extra fields are opt-in (`encode(..., extra=True)`)

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。

    每行:
      0 date, 1 user_id, 2 video_id, 3 author_id, 4 tab, 5 duration_ms,
      6 long_view, 7 time_ms, 8 hourmin, 9 is_click, 10 play_time_ms
    前 7 项下标与 starter kit 原版一致（pop / evaluate 仍可用）。
    """
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             int(r['time_ms']), r['hourmin'],
                             1 if r['is_click'] != '0' else 0,
                             float(r['play_time_ms'])))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def item_pop_map(splits, prior=20.0):
    """Train-only smoothed long_view rate per video (item-side; first-order helps within-user rank)."""
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1
        pos[x[2]] += x[6]
    gmean = (sum(pos.values()) / sum(imp.values())) if imp else 0.0
    pop = {}
    for v, n in imp.items():
        pop[v] = (pos[v] + prior * gmean) / (n + prior)
    return pop, gmean


def pop_scores_for_split(splits, name, prior=20.0):
    pop, gmean = item_pop_map(splits, prior=prior)
    return np.array([pop.get(x[2], gmean) for x in splits[name]], dtype=np.float32)


def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def _causal_last_videos(splits):
    """For each impression, previous video / previous long_view video of the same user (time_ms-strict)."""
    events = []
    for name in ('train', 'valid', 'test'):
        for i, x in enumerate(splits[name]):
            events.append((x[7], name, i, x[1], x[2], x[6]))
    events.sort()
    last_vid, last_pos = {}, {}
    out_vid = {n: ['NONE'] * len(splits[n]) for n in splits}
    out_pos = {n: ['NONE'] * len(splits[n]) for n in splits}
    for _, name, i, u, v, y in events:
        out_vid[name][i] = last_vid.get(u, 'NONE')
        out_pos[name][i] = last_pos.get(u, 'NONE')
        last_vid[u] = v
        if y:
            last_pos[u] = v
    return out_vid, out_pos


def encode(splits, extra=False):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(fields))，以及 field_dims。

    extra=True 加入 hour / train-pop / 因果序列（官方未测过的方向）。
    extra=False 复现官方 5 域 FM。
    """
    fields = BASE_FIELDS + EXTRA_FIELDS if extra else BASE_FIELDS
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    pop, _gmean = item_pop_map(splits)
    pop_edges = _bucket_edges(list(pop.values())) if pop else np.array([0.5])
    last_vid, last_pos = _causal_last_videos(splits) if extra else (None, None)

    def hour_bucket(x):
        try:
            return str(int(str(x[8]).zfill(4)[:2]))
        except (TypeError, ValueError):
            return 'UNK'

    def pop_bucket(x):
        r = pop.get(x[2])
        if r is None:
            return 'UNK'
        return str(int(np.searchsorted(pop_edges, r)))

    def raw(name, i, x):
        f = [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5])))]
        if extra:
            f += [hour_bucket(x), pop_bucket(x), last_vid[name][i], last_pos[name][i]]
        return f

    vocabs = [dict() for _ in fields]
    for i, x in enumerate(tr):
        for j, v in enumerate(raw('train', i, x)):
            if v not in vocabs[j]:
                vocabs[j][v] = len(vocabs[j])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for j, v in enumerate(raw(name, n, x)):
                X[n, j] = vocabs[j].get(v, unk[j]) + offsets[j]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    enc['_fields'] = fields
    return enc, int(sum(field_dims))


def _smoothed(pos, n, gmean, prior=20.0):
    return (pos + prior * gmean) / (n + prior)


def rank_components(splits, prior=20.0):
    """Train-only ranking priors + causal same-author flag.

    These are not the failed CWM user buckets. They only use train labels / counts,
    except same_author which is time_ms-causal (no future leak).
    Returns {split: {name: np.ndarray}}.
    """
    item_pos = collections.Counter(); item_imp = collections.Counter()
    item_click = collections.Counter(); item_watch = collections.Counter()
    author_pos = collections.Counter(); author_imp = collections.Counter()
    ua_pos = collections.Counter(); ua_imp = collections.Counter()
    uv_pos = collections.Counter()

    for x in splits['train']:
        u, v, a = x[1], x[2], x[3]
        y, c, play, dur = x[6], x[9], x[10], max(x[5], 1.0)
        item_imp[v] += 1; item_pos[v] += y; item_click[v] += c
        item_watch[v] += min(play / dur, 2.0)
        author_imp[a] += 1; author_pos[a] += y
        ua_imp[u, a] += 1; ua_pos[u, a] += y
        uv_pos[u, v] += y

    g_item = (sum(item_pos.values()) / sum(item_imp.values())) if item_imp else 0.0
    g_auth = (sum(author_pos.values()) / sum(author_imp.values())) if author_imp else 0.0
    g_click = (sum(item_click.values()) / sum(item_imp.values())) if item_imp else 0.0
    g_watch = (sum(item_watch.values()) / sum(item_imp.values())) if item_imp else 0.0
    g_ua = g_item

    events = []
    for name in ('train', 'valid', 'test'):
        for i, x in enumerate(splits[name]):
            events.append((x[7], name, i, x[1], x[3]))
    events.sort()
    last_author = {}
    same_author = {n: np.zeros(len(splits[n]), dtype=np.float32) for n in splits}
    for _, name, i, u, a in events:
        same_author[name][i] = 1.0 if last_author.get(u) == a else 0.0
        last_author[u] = a

    out = {}
    for name, rws in splits.items():
        n = len(rws)
        item_pop = np.empty(n, dtype=np.float32)
        author_pop = np.empty(n, dtype=np.float32)
        item_clk = np.empty(n, dtype=np.float32)
        item_w = np.empty(n, dtype=np.float32)
        ua = np.empty(n, dtype=np.float32)
        uv = np.empty(n, dtype=np.float32)
        for i, x in enumerate(rws):
            u, v, a = x[1], x[2], x[3]
            ni = item_imp[v]
            item_pop[i] = _smoothed(item_pos[v], ni, g_item, prior) if ni else g_item
            item_clk[i] = _smoothed(item_click[v], ni, g_click, prior) if ni else g_click
            item_w[i] = _smoothed(item_watch[v], ni, g_watch, prior) if ni else g_watch
            na = author_imp[a]
            author_pop[i] = _smoothed(author_pos[a], na, g_auth, prior) if na else g_auth
            nua = ua_imp[u, a]
            ua[i] = _smoothed(ua_pos[u, a], nua, g_ua, prior) if nua else g_ua
            uv[i] = float(uv_pos[u, v])
        out[name] = {
            'item_pop': item_pop,
            'author_pop': author_pop,
            'item_click': item_clk,
            'item_watch': item_w,
            'user_author': ua,
            'user_video': uv,
            'same_author': same_author[name],
        }
    return out
