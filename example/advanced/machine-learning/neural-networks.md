# Neural Networks and Deep Learning

## Mathematical Foundations

### Perceptron

The basic perceptron computes: $y = f(\sum_{i=1}^{n} w_i x_i + b)$

Where $f$ is the activation function, commonly:
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **ReLU**: $\text{ReLU}(x) = \max(0, x)$
- **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

### Backpropagation

**Chain Rule**: $\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l+1)}} \cdot \frac{\partial a_j^{(l+1)}}{\partial z_j^{(l+1)}} \cdot \frac{\partial z_j^{(l+1)}}{\partial w_{ij}^{(l)}}$

**Gradient Update**: $w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta \frac{\partial L}{\partial w_{ij}^{(l)}}$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ConvolutionalNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ConvolutionalNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming input size is 32x32, after 3 pooling operations: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_synthetic_data(n_samples=1000, n_features=10, n_classes=3):
    """Create synthetic classification dataset"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear relationships
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 0 if X[i, 2] > 0 else 1
        else:
            y[i] = 2
    
    return torch.FloatTensor(X), torch.LongTensor(y.astype(int))

# Generate data
X, y = create_synthetic_data(1000, 10, 3)

# Split data
train_size = 800
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create and train model
model = NeuralNetwork(input_size=10, hidden_sizes=[64, 32], output_size=3)
trainer = Trainer(model)

print("Training Neural Network...")
trainer.train(train_loader, val_loader, num_epochs=50)
trainer.plot_history()
```

## Advanced Architectures

### Transformer Model

```python
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### Generative Adversarial Network (GAN)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GANTrainer:
    def __init__(self, generator, discriminator, latent_dim, device='cpu'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    def train_step(self, real_imgs):
        batch_size = real_imgs.shape[0]
        
        # Adversarial ground truths
        valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
        fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)
        
        # Configure input
        real_imgs = real_imgs.to(self.device)
        
        # -----------------
        #  Train Generator
        # -----------------
        
        self.optimizer_G.zero_grad()
        
        # Sample noise as generator input
        z = torch.FloatTensor(batch_size, self.latent_dim).normal_(0, 1).to(self.device)
        
        # Generate a batch of images
        gen_imgs = self.generator(z)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        self.optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        return g_loss.item(), d_loss.item()

# Example usage
latent_dim = 100
img_shape = (1, 28, 28)  # For MNIST

generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)
gan_trainer = GANTrainer(generator, discriminator, latent_dim)
```

## Optimization Techniques

### Advanced Optimizers

The **Adam optimizer** combines momentum and adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Learning Rate Scheduling

```python
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_steps=1000, d_model=512):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Cosine annealing with warm restarts
class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                               (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        self.T_cur += 1
```

## Model Interpretability

### Gradient-based Attribution

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx=None):
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

# SHAP (SHapley Additive exPlanations) for feature importance
def compute_shap_values(model, background_data, test_data):
    """
    Compute SHAP values for neural network
    """
    import shap
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background_data)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(test_data)
    
    return shap_values

# Integrated Gradients
class IntegratedGradients:
    def __init__(self, model):
        self.model = model
    
    def generate_baseline(self, input_tensor, baseline_type='zero'):
        if baseline_type == 'zero':
            return torch.zeros_like(input_tensor)
        elif baseline_type == 'random':
            return torch.randn_like(input_tensor)
        elif baseline_type == 'blur':
            # Apply Gaussian blur
            return self.apply_blur(input_tensor)
    
    def compute_gradients(self, input_tensor, target_class):
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        target_score = output[0, target_class]
        target_score.backward()
        
        return input_tensor.grad.detach()
    
    def compute_integrated_gradients(self, input_tensor, target_class, 
                                   baseline=None, steps=50):
        if baseline is None:
            baseline = self.generate_baseline(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        
        # Compute gradients for interpolated inputs
        gradients = []
        for interpolated in interpolated_inputs:
            grad = self.compute_gradients(interpolated, target_class)
            gradients.append(grad)
        
        # Average the gradients and scale by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        return integrated_gradients
```

## Transfer Learning and Fine-tuning

```python
import torchvision.models as models

class TransferLearningModel:
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        if self.model_name == 'resnet50':
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'vgg16':
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
        elif self.model_name == 'efficientnet':
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=10):
        """Unfreeze top num_layers for fine-tuning"""
        # Get all parameters
        all_params = list(self.model.parameters())
        
        # Unfreeze last num_layers
        for param in all_params[-num_layers:]:
            param.requires_grad = True
    
    def get_model(self):
        return self.model

# Progressive unfreezing strategy
class ProgressiveUnfreezingTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = []
    
    def train_phase(self, num_epochs, learning_rate, unfreeze_layers=None):
        if unfreeze_layers is not None:
            self.unfreeze_layers(unfreeze_layers)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = 100. * correct / len(self.val_loader.dataset)
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss / len(self.train_loader),
                'val_loss': val_loss / len(self.val_loader),
                'val_accuracy': accuracy
            })
            
            print(f'Epoch {epoch}: Val Loss: {val_loss/len(self.val_loader):.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
    
    def unfreeze_layers(self, layer_names):
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True
```