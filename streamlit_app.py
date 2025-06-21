import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters (must match training script)
LATENT_DIM = 100
IMAGE_SIZE = 28

class Generator(nn.Module):
    """Generator network for creating MNIST digits"""
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Generator layers
        self.model = nn.Sequential(
            # Input: latent_dim * 2 (noise + label embedding)
            nn.Linear(latent_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_emb], dim=1)
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    try:
        generator = Generator(LATENT_DIM).to(device)
        generator.load_state_dict(torch.load('generator.pth', map_location=device))
        generator.eval()
        return generator
    except FileNotFoundError:
        st.error("⚠️ Model file 'generator.pth' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

def generate_digit_images(generator, digit, num_images=5):
    """Generate multiple images of a specific digit"""
    with torch.no_grad():
        # Create labels for the specified digit
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        # Generate random noise
        noise = torch.randn(num_images, LATENT_DIM).to(device)
        
        # Generate images
        fake_imgs = generator(noise, labels)
        fake_imgs = fake_imgs.cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2.0
        
        return fake_imgs

def create_image_grid(images):
    """Create a grid of images for display"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.patch.set_facecolor('white')
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Generated #{i+1}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return img_base64

# Main app
def main():
    # Header
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("---")
    
    # Description
    st.markdown("""
    This web application generates handwritten digits (0-9) using a trained Conditional GAN model.
    Select a digit below and click **Generate** to create 5 unique images of that digit.
    """)
    
    # Load model
    generator = load_model()
    
    if generator is None:
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Digit selection
        selected_digit = st.selectbox(
            "Select digit to generate:",
            options=list(range(10)),
            index=0,
            help="Choose which digit (0-9) you want to generate"
        )
        
        st.markdown("---")
        
        # Generate button
        generate_button = st.button(
            "🎲 Generate Images",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Info")
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Latent Dimension:** {LATENT_DIM}")
        st.markdown(f"**Image Size:** {IMAGE_SIZE}×{IMAGE_SIZE}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Generated Images for Digit: {selected_digit}")
        
        # Generate images when button is clicked
        if generate_button or 'last_digit' not in st.session_state or st.session_state.last_digit != selected_digit:
            with st.spinner("🔄 Generating images..."):
                try:
                    # Generate images
                    images = generate_digit_images(generator, selected_digit, 5)
                    
                    # Store in session state
                    st.session_state.generated_images = images
                    st.session_state.last_digit = selected_digit
                    
                    # Create and display image grid
                    fig = create_image_grid(images)
                    st.pyplot(fig)
                    
                    # Success message
                    st.success(f"✅ Successfully generated 5 images of digit {selected_digit}!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating images: {str(e)}")
        
        # Display previously generated images if available
        elif 'generated_images' in st.session_state:
            fig = create_image_grid(st.session_state.generated_images)
            st.pyplot(fig)
    
    with col2:
        st.header("ℹ️ About")
        st.markdown("""
        **Model Architecture:**
        - Conditional GAN (cGAN)
        - Generator: 4-layer MLP
        - Discriminator: 4-layer MLP
        - Trained on MNIST dataset
        
        **Features:**
        - Generates 28×28 grayscale images
        - Conditioned on digit class (0-9)
        - Each generation produces unique variations
        - MNIST-style handwritten digits
        """)
        
        # Statistics
        if 'generated_images' in st.session_state:
            st.markdown("### 📈 Current Generation")
            images = st.session_state.generated_images
            st.metric("Images Generated", "5")
            st.metric("Selected Digit", st.session_state.last_digit)
            st.metric("Image Resolution", "28×28")
            
            # Show individual images
            st.markdown("#### Individual Images")
            for i in range(5):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    fig_small, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(images[i, 0], cmap='gray', vmin=0, vmax=1)
                    ax.axis('off')
                    st.pyplot(fig_small)
                    plt.close(fig_small)
                with col_b:
                    st.markdown(f"**Image #{i+1}**")
                    st.markdown(f"Shape: {images[i, 0].shape}")
                    st.markdown(f"Min/Max: {images[i, 0].min():.3f}/{images[i, 0].max():.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and PyTorch | MNIST Digit Generator</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
