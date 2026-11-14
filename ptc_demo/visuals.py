"""Presentation view of measured runs; no API calls or new benchmark arithmetic."""
from html import escape

from .benchmark import summarize


def benchmark_dashboard(datasets, *, saved):
    """Return a Plotly figure with a dataset selector and honest failure labels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not datasets or any(not p.get('runs') for p in datasets.values()):
        raise ValueError('Supply at least one nonempty benchmark')
    colors = {'standard': '#7EA6FF', 'ptc': '#39D5B1'}
    names = {'standard': 'Standard', 'ptc': 'Hosted PTC'}
    fig = make_subplots(rows=3, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'xy'}, {'type': 'xy'}], [{'colspan': 2}, None]],
        row_heights=[0.16, 0.38, 0.46], vertical_spacing=0.13,
        subplot_titles=('', '', 'MEDIAN TOTAL TOKENS', 'MEDIAN ELAPSED TIME',
                        'EVERY RUN · ACTUAL EXECUTION ORDER'))
    buttons = []
    for index, (label, payload) in enumerate(datasets.items()):
        runs = payload['runs']
        summary = summarize(runs)
        medians = summary['medians']
        if set(medians) != set(names):
            raise ValueError('Each dataset needs both standard and PTC runs')
        visible = index == 0
        for column, mode in enumerate(names, 1):
            stats = medians[mode]
            fig.add_trace(go.Indicator(mode='number', value=stats['correct_runs'],
                number=dict(suffix=f" / {stats['runs']}", font=dict(size=40, color=colors[mode])),
                title=dict(text=f'{names[mode]} · exact answers', font=dict(size=16)),
                visible=visible), row=1, col=column)
        for column, metric in enumerate(('total_tokens', 'elapsed_seconds'), 1):
            values = [medians[mode][metric] for mode in names]
            text = ['Unknown' if v is None else (f'{v:,.0f}' if column == 1 else f'{v:.2f}s')
                    for v in values]
            fig.add_trace(go.Bar(x=list(names.values()), y=values, text=text,
                textposition='outside', cliponaxis=False, marker_color=list(colors.values()),
                marker_pattern_shape=['/' if medians[m]['correct_runs'] < medians[m]['runs'] else ''
                                      for m in names],
                hovertemplate='%{x}<br>Median: %{text}<extra></extra>',
                visible=visible, showlegend=False), row=2, col=column)
        status = [('Correct' if r['correctness'] else 'Incorrect / failed') +
                  ('' if r['usage_complete'] else ' · usage incomplete') for r in runs]
        details = [[status[i], r['usage'].get('input_tokens'), r['usage'].get('cached_input_tokens'),
                    r['usage'].get('output_tokens'), r['usage'].get('reasoning_tokens'),
                    r['model_requests'], r['tool_invocations'], r['elapsed_seconds']]
                   for i, r in enumerate(runs)]
        fig.add_trace(go.Bar(
            x=[f"{i+1}. {names[r['approach']]} · R{r.get('repetition', 1)}" for i, r in enumerate(runs)],
            y=[r['usage'].get('total_tokens') for r in runs], customdata=details,
            text=[f"{r['usage']['total_tokens']:,}" if r['usage'].get('total_tokens') is not None
                  else 'Unknown usage' for r in runs],
            textposition='outside', cliponaxis=False,
            marker=dict(color=[colors[r['approach']] for r in runs],
                        pattern_shape=['' if r['correctness'] and r['usage_complete'] else '/' for r in runs]),
            hovertemplate=('%{x}<br><b>%{customdata[0]}</b><br>Total tokens: %{y:,}'
                '<br>Input: %{customdata[1]} · cached subset: %{customdata[2]}'
                '<br>Output: %{customdata[3]} · reasoning subset: %{customdata[4]}'
                '<br>API requests: %{customdata[5]} · tools: %{customdata[6]}'
                '<br>Elapsed: %{customdata[7]:.2f}s<extra></extra>'),
            visible=visible, showlegend=False), row=3, col=1)
        reduction = summary['token_reduction_percent']
        insight = ('Savings withheld — correctness, usage or model comparability check failed'
                   if reduction is None else f'Median token reduction: {reduction:.2f}% · not a cost or latency claim')
        source = 'SAVED RESULTS — NOT LIVE' if saved else 'LIVE SESSION RESULTS'
        title = dict(text=f'<b>Code Mode / PTC</b> · {escape(label)}<br>'
            f'<span style="font-size:13px">{source} · {escape(payload.get("model", ""))} · '
            f'{escape(payload.get("created_at", ""))}<br>{insight}</span>', x=0.05, y=0.98)
        buttons.append(dict(label=label, method='update', args=[
            {'visible': [j // 5 == index for j in range(len(datasets) * 5)]},
            {'title': title, 'xaxis3.autorange': True, 'yaxis.autorange': True,
             'yaxis2.autorange': True, 'yaxis3.autorange': True}]))
        if visible:
            fig.update_layout(title=title)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#101827', plot_bgcolor='#182337',
        font=dict(family='Arial, sans-serif', size=14, color='#E7EDF8'), height=920,
        margin=dict(l=75, r=45, t=195, b=95), bargap=0.42,
        updatemenus=[dict(buttons=buttons, x=1, y=1.08, xanchor='right', yanchor='top',
                         bgcolor='#26364E', bordercolor='#64748B')],
        hoverlabel=dict(bgcolor='#243249', font_size=13))
    fig.update_yaxes(rangemode='tozero', gridcolor='#2C3950', zeroline=False)
    fig.update_yaxes(title_text='Tokens', row=2, col=1)
    fig.update_yaxes(title_text='Seconds', row=2, col=2)
    fig.update_yaxes(title_text='Reported tokens', row=3, col=1)
    fig.update_xaxes(showgrid=False, fixedrange=False)
    fig.add_annotation(x=0, y=-0.13, xref='paper', yref='paper', showarrow=False, xanchor='left',
        text='Hatched = incorrect / failed or incomplete usage. All runs retained. '
             'Unknown usage is not zero.<br>Hover for cached / reasoning subsets; neither is added twice. '
             'Medians with missing usage are withheld.', font=dict(size=12, color='#B8C6DC'))
    return fig
