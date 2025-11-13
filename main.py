import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import os
import torch
import torch.nn.functional as F

MODEL_LIST = [
    # Llama family.
    'Llama-2-7b-hf',
    'Llama-2-7b-chat-hf',
    'Llama-2-13b-hf',
    'Llama-2-13b-chat-hf',
    'Llama-2-70b-chat-hf',

    # Qwen family.
    'Qwen2.5-0.5B',
    'Qwen2.5-1.5B-Instruct',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-14B',
    'Qwen2.5-32B',

    # Gemma family.
    'gemma-2-2b',
    'gemma-2-9b',
    'gemma-2-27b',

    # Shield gemma family.
    'shieldgemma-2b',
    'shieldgemma-9b',
    'shieldgemma-27b',
]

# BASE_DIR = os.path.join("", "post_process")
BASE_DIR = os.path.join(os.getcwd(), "cosine_sim_values_v2" )
TSNE_DIR = os.path.join(os.getcwd(), "activations_postprocess") 

# Page configuration
st.set_page_config(
    page_title="Model Activation Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============== UTILS ==================
def get_cosine_similarity(activation_path):
    """
    Calculate cosine similarity between different harm categories.
    """
    
    df_path = os.path.join(activation_path, "cosine_sim.csv")
    dataFrame = pd.read_csv(df_path, index_col = 0)
    
    return dataFrame


def plot_cosine_similarity_heatmap(df, title="Cosine Similarity Heatmap"):
    """
    Create an interactive Plotly heatmap with cosine similarity values displayed.
    """
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdBu_r',  # Red-Blue reversed (red for high similarity)
        zmid=0,  # Center the colorscale at 0
        text=np.round(df.values, 3),  # Display values rounded to 3 decimals
        texttemplate='%{text}',  # Show text on heatmap
        textfont={"size": 10},
        colorbar=dict(
            title="Cosine Similarity",
            tickmode="linear",
            tick0=0,
            dtick=0.2
        ),
        hoverongaps=False,
        hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title="Harm Category",
        yaxis_title="Harm Category",
        width=900,
        height=800,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange='reversed')  # To match matrix orientation
    )
    
    return fig


def display_dataframe_stats(df):
    """
    Display statistics about the cosine similarity matrix.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    # Exclude diagonal (self-similarity)
    mask = np.ones(df.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diagonal_values = df.values[mask]
    
    with col1:
        st.metric("Mean Similarity", f"{off_diagonal_values.mean():.4f}")
        pass
    
    with col2:
        st.metric("Max Similarity", f"{off_diagonal_values.max():.4f}")
        pass
    
    with col3:
        st.metric("Min Similarity", f"{off_diagonal_values.min():.4f}")
        pass
    
    with col4:
        st.metric("Std Dev", f"{off_diagonal_values.std():.4f}")
        pass

def generate_tsne_plot(activation_path, model_name, token_strategy, perplexity=30, random_state=42):
    """
    Generate interactive t-SNE visualization from activation tensors.
    
    Parameters:
    -----------
    activation_path : str
        Path to directory containing activation tensors
    model_name : str
        Name of the model for the plot title
    token_strategy : str
        Token strategy used (e.g., 'first_5_tokens')
    perplexity : int
        t-SNE perplexity parameter (default: 30)
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    df_viz : pd.DataFrame
        DataFrame with t-SNE results and metadata
    """
    
    if not os.path.exists(activation_path):
        st.error(f"Path does not exist: {activation_path}")
        return None, None
    
    # Load all activation files
    harm_categories = sorted([f for f in os.listdir(activation_path) if f.endswith('.pt')])
    
    if not harm_categories:
        st.warning(f"No .pt files found in: {activation_path}")
        return None, None
    
    all_data = []
    all_labels = []
    category_names = []
    
    for category_file in harm_categories:
        category_name = category_file.split('.')[0]
        category_path = os.path.join(activation_path, category_file)
        
        try:
            tensor = torch.load(category_path)
            
            # tensor shape: [prompts, layers, positions, hidden_dims]
            # Reduce: mean across positions (dim=2), then mean across layers (dim=1)
            # Result: [prompts, hidden_dims]
            if tensor.dim() == 4:
                flat_tensor = tensor.mean(dim=2).mean(dim=1)  # [prompts, hidden_dims]
            elif tensor.dim() == 3:
                flat_tensor = tensor.mean(dim=1)  # [prompts, hidden_dims]
            elif tensor.dim() == 2:
                flat_tensor = tensor  # Already [prompts, hidden_dims]
            else:
                st.error(f"Unexpected tensor shape for {category_name}: {tensor.shape}")
                continue
            
            num_prompts = flat_tensor.shape[0]
            
            all_data.append(flat_tensor.detach().cpu().numpy())
            all_labels.extend([category_name] * num_prompts)
            category_names.append(category_name)
            
        except Exception as e:
            st.error(f"Error loading {category_file}: {e}")
            continue
    
    if not all_data:
        st.warning("No valid tensors loaded for t-SNE")
        return None, None
    
    # Stack all data
    all_tensor = np.vstack(all_data)
    
    # Perform t-SNE
    with st.spinner(f"Computing t-SNE (perplexity={perplexity})..."):
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, verbose=0)
        tsne_results = tsne.fit_transform(all_tensor)
    
    # Create DataFrame for visualization
    df_viz = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'category': all_labels,
        'point_index': range(len(tsne_results))
    })
    
    # Generate distinct colors for categories
    num_categories = len(category_names)
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    colors = colors[:num_categories]

    colors = colors = ['red','blue','green','orange','purple','black','pink','brown','turquoise','violet','grey','yellow']
    
    # Create interactive plot
    fig = px.scatter(
        df_viz, 
        x='x', 
        y='y', 
        color='category',
        hover_data=['point_index'],
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
        color_discrete_sequence=colors,
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')))
    
    fig.update_layout(

        height = 800,
        width = 1000,

        # White background
        plot_bgcolor='white',
        paper_bgcolor='white',

        title = dict(
            text=f't-SNE Visualization: {model_name} ({token_strategy})',
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='black')  # Set title color to black
        ),
        
        # Clean grid lines
        xaxis=dict(
            title=dict(text='t-SNE Component 1', font=dict(color='black', size=14)),  # ADD THIS
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            tickfont=dict(color='black', size=12),
        ),
        yaxis=dict(
            title=dict(text='t-SNE Component 2', font=dict(color='black', size=14)),  # ADD THIS
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            tickfont=dict(color='black', size=12),
        ),
        
        # Font styling
        font = dict(
            family="Arial, sans-serif",
            size=12,
            color='black'
        ),
        
        # Legend positioning
        legend = dict(
            bgcolor='white',
            bordercolor='lightgray',
            borderwidth=1,
            font=dict(size=10, color='black')
        ))

    # Update marker styling
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.8,
            line=dict(width=0)  # Remove white borders around points
        ))
    
    return fig, df_viz


def plot_tsne_with_options(activation_path, model_name, token_strategy):
    """
    Wrapper function to display t-SNE plot with interactive options in Streamlit.
    """
    
    st.markdown("#### t-SNE plots visualization.")

    # Generate plot
    fig, df_viz = generate_tsne_plot(
        activation_path,
        model_name,
        token_strategy,
        perplexity=30,
        random_state=42
    )
    
    if fig is not None:
        st.plotly_chart(fig)
        
    
    return fig, df_viz

# Main App
def main():
    st.header('Model Activation Analysis Dashboard')
    st.markdown("Analyze model activations across different strategies and layers with interactive visualizations")

    # Methodology explanation
    with st.expander("📖 **Methodology: How are directions calculated?**", expanded=False):
        st.markdown("""
        <div class="methodology-box">
        <h4>Direction Calculation from Activation Values</h4>
        <p>Hidden state activations are collected for each layer in the model. The dimensions of the activation 
        array is of the shape <strong>[layer_num, num_prompts, num_tokens, hidden_dims]</strong>.</p>
        
        <p>The number of prompts across all categories is <strong>50</strong> and <code>layer_num</code> varies with each model. 
        <code>num_tokens</code> are the number of tokens we collect. For the <code>first_5_tokens</code> strategy, 
        it is 5 while for <code>first_tok</code>, it is 1 and so on.</p>
        
        <p><strong>We calculate the direction for each harm category by:</strong></p>
        <ol>
            <li>Averaging across prompts and tokens</li>
            <li>Subtracting the direction tensor with the harmless direction tensor. This is 
                consistent with the literature of using difference-in-means to isolate a concept direction.</li>
            <li>Flattening the residual tensor</li>
        </ol>
                    
        <p> The reasoning behind averaging across prompts and tokens is to average information across all prompts and tokens but
            preserve the information on each layer. <p/>

        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("**Transformation Pipeline:**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Vertical model selection using radio buttons
        selected_model = st.radio(
            "Select Model",
            MODEL_LIST,
            index=0,
        )
        
        st.info(f"Currently analyzing: **{selected_model}**")
        
        st.markdown("---")
        
        # Additional options
        st.subheader("Visualization Options")
        colorscale = st.selectbox(
            "Color Scheme",
            ['RdBu_r', 'Viridis', 'Plasma', 'Inferno', 'Cividis', 'Turbo'],
            index=0
        )

    # Main tabs remain horizontal
    tab1, tab2, tab3, tab4 = st.tabs([
        "First 5 Tokens", 
        "First Token", 
        "Last 5 Tokens",
        "Last Token",
    ])

    # Function to display content in each tab
    def display_tab_content(token_strategy):
        st.markdown(f"### Analysis for: **{selected_model}** | Strategy: **{token_strategy}**")
        
        activations_path = os.path.join(BASE_DIR, selected_model, token_strategy)
        
        with st.spinner(f"Loading and calculating cosine similarities for {token_strategy}..."):
            cosine_similarity_df = get_cosine_similarity(activations_path)
        
        if cosine_similarity_df is not None:
            # Display statistics
            st.markdown("#### Similarity Statistics")
            display_dataframe_stats(cosine_similarity_df)
            
            st.markdown("---")
            
            # Display heatmap
            st.markdown("#### Cosine Similarity Heatmap")
            fig = plot_cosine_similarity_heatmap(
                cosine_similarity_df,
                title=f"Cosine Similarity - {selected_model} ({token_strategy})"
            )
            
            # Update colorscale based on user selection
            fig.data[0].colorscale = colorscale
            
            st.plotly_chart(fig, width='stretch')

            st.markdown("---")

        else:
            st.error("Could not load cosine similarity data. Please check the data path.")

    with tab1:
        display_tab_content("first_5_tokens")

        plot_tsne_with_options(
            activation_path= os.path.join(TSNE_DIR, selected_model, "first_5_tokens"),
            model_name = selected_model,
            token_strategy="first_5_tokens"
        )

    with tab2:
        display_tab_content("first_token")

        plot_tsne_with_options(
            activation_path= os.path.join(TSNE_DIR, selected_model, "first_token"),
            model_name = selected_model,
            token_strategy="first_token"
        )

    with tab3:
        display_tab_content("last_5_tokens")

        plot_tsne_with_options(
            activation_path= os.path.join(TSNE_DIR, selected_model, "last_5_tokens"),
            model_name = selected_model,
            token_strategy="last_5_tokens"
        )

    with tab4:
        display_tab_content("last_token")

        plot_tsne_with_options(
            activation_path= os.path.join(TSNE_DIR, selected_model, "last_token"),
            model_name = selected_model,
            token_strategy="last_token"
        )


if __name__ == "__main__":
    main()