
import plotly.graph_objects as go

def plot_field(xlim_l = -0.5, xlim_r = 120.5, ablos = None, ygtline = None):

    shapes = []
    for yard in range(20,110,10):
        line = go.layout.Shape(
        type = 'line',
        x0 = yard, y0 = 0, x1 = yard, y1 = 53.3,
        line = dict(color = 'grey', width = 0.5))
        shapes.append(line)

    ez1 = go.layout.Shape(
        type = "rect", x0 = 0, y0 = 0, x1 = 10,y1 = 53.3,
        line = dict(color = 'darkgrey', width = 1),  
        fillcolor = 'lightgrey'
    )
    ez2 = go.layout.Shape(
        type = "rect", x0 = 110, y0 = 0, x1 = 120,y1 = 53.3,
        line = dict(color = 'darkgrey', width = 1),  
        fillcolor = 'lightgrey'
    )
    
    if ablos:
        losline = go.layout.Shape(
            type = 'line',
            x0 = ablos, y0 = 0, x1 = ablos, y1 = 53.3,
            line = dict(color = 'RGB(255,215,0)', width = 1.3)
        )
        shapes.append(losline)
        
    if ygtline:
        ygtline = go.layout.Shape(
            type = 'line',
            x0 = ygtline, y0 = 0, x1 = ygtline, y1 = 53.3,
            line = dict(color = 'RGB(255,215,0)', width = 1.3, dash='dash')
        )
        shapes.append(ygtline)

    for i in range(11, 110, 1):
        lines = [
            {'type': 'line', 'x0': i, 'y0': 0.2, 'x1': i, 'y1': 0.7, 'line': dict(color='darkgrey', width=0.5)},
            {'type': 'line', 'x0': i, 'y0': 22.91, 'x1': i, 'y1': 23.57, 'line': dict(color='darkgrey', width=0.5)},
            {'type': 'line', 'x0': i, 'y0': 29.73, 'x1': i, 'y1': 30.39, 'line': dict(color='darkgrey', width=0.5)},
            {'type': 'line', 'x0': i, 'y0': 53, 'x1': i, 'y1': 52.5, 'line': dict(color='darkgrey', width=0.5)}
        ]
        shapes.extend(lines)

    field_shape = go.layout.Shape(
        type = "rect", x0 = 0, y0 = 0, x1 = 120, y1 = 53.3,
        line = dict(color = 'black', width = 1),
    )
    shapes.append(field_shape)
    
    layout = go.Layout(
        xaxis=dict(range=[xlim_l, xlim_r], constrain='range', showticklabels=False, showgrid=False),
        yaxis=dict(range=[-0.5, 54], scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False),
        shapes = shapes,
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig = go.Figure(layout = layout)
    
    fig.add_shape(ez1, layer='below')
    fig.add_shape(ez2, layer='below')

    for i in range(10, 100, 10):
        if i > 50:
            yard = 100-i
        else:
            yard = i
        text1 = go.layout.Annotation(
            xref = 'x', yref = 'y',
            x = i+9, y = 5, text = yard, font=dict(color='grey', size=8),
            showarrow=False)   
        text2 = go.layout.Annotation(
            xref = 'x', yref = 'y', textangle = 180,
            x = i+9, y = 48.3, text = yard, font=dict(color='grey', size=8),
            showarrow=False)    
        fig.add_annotation(text1)
        fig.add_annotation(text2)

    return fig
