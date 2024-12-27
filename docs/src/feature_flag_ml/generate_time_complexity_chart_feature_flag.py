import plotly.graph_objects as go
import numpy as np
import os

def generate_complexity_comparison():
    # Generate data points
    n = np.arange(1, 101, 5)
    linear = n
    logarithmic = np.log2(n) * 5
    
    # Create the main figure
    fig = go.Figure()
    
    # Add shaded area for k-NN faster region
    n_knn = n[n < 25]
    linear_knn = linear[n < 25]
    
    fig.add_trace(go.Scatter(
        x=list(n_knn) + list(n_knn[::-1]),
        y=list(linear_knn) + [0] * len(n_knn),
        fill='tozeroy',
        fillcolor='rgba(240,240,255,0.6)',
        line=dict(width=0),
        showlegend=True,
        name='k-NN Faster Region'
    ))
    
    # Add main lines
    fig.add_trace(go.Scatter(
        x=n,
        y=linear,
        mode='lines',
        name='O(n) - Linear Search (k-NN)',
        line=dict(color='#ff4d4d', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=n,
        y=logarithmic,
        mode='lines',
        name='O(log n) - Approximate Search (ANN)',
        line=dict(color='#4d4dff', width=2)
    ))
    
    # Find crossover point
    crossover_idx = np.argmin(np.abs(linear - logarithmic))
    crossover_x = n[crossover_idx]
    crossover_y = linear[crossover_idx]
    
    # Add crossover point
    fig.add_trace(go.Scatter(
        x=[crossover_x],
        y=[crossover_y],
        mode='markers',
        marker=dict(size=10, color='#8884d8'),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Time Complexity: k-NN vs ANN Search',
        xaxis_title='Dataset Size (n)',
        yaxis_title='Time',
        template='plotly_white',
        width=800,
        height=600,
        margin=dict(t=100, b=100),
        annotations=[
            # Add key insights
            dict(
                x=0.02,
                y=-0.25,
                xref='paper',
                yref='paper',
                text='<b>Key Insights:</b><br>' +
                     '• k-NN Advantage Zone (Highlighted): For smaller datasets, k-NN\'s simple linear search is often faster<br>' +
                     '• Crossover Point: Around this point, ANN structures begin to show their advantage<br>' +
                     '• ANN Advantage Zone: Beyond crossover, ANN\'s logarithmic complexity provides better performance',
                showarrow=False,
                align='left'
            )
        ]
    )
    
    os.makedirs('charts', exist_ok=True)
    
    fig.write_html('charts/complexity_comparison.html')
    print("Chart has been saved to charts/complexity_comparison.html")

if __name__ == "__main__":
    generate_complexity_comparison()
