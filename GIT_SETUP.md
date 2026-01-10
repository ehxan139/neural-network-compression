# Git Setup Commands for Neural Network Compression

This file contains the Git commands to initialize and push the project to GitHub.

## Prerequisites

1. Install Git: https://git-scm.com/downloads
2. Configure Git with your credentials (if not already done)
3. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Repository name: `neural-network-compression`
   - Description: "Production-ready model compression toolkit: Knowledge distillation + Gradual pruning + Low-rank factorization. Up to 95% size reduction with 98%+ accuracy retention."
   - Set to Public
   - Do NOT initialize with README (we have one)
   - Click "Create repository"

## Commands to Execute

### Using HTTPS (Recommended)

```bash
# Navigate to project directory
cd "C:\Users\ehsan\OneDrive\github_ehxan139\neural-network-compression"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Neural network compression toolkit

- Knowledge distillation (80-90% compression, 98%+ accuracy)
- Gradual pruning (85-95% sparsity with polynomial schedule)
- Low-rank factorization (SVD-based layer decomposition)
- VGG-19, ResNet-50, and mobile-optimized student architectures
- Comprehensive benchmarking on CIFAR-10
- Production deployment guides for mobile/edge
- Jupyter notebooks with detailed examples
"

# Set main branch
git branch -M main

# Add remote repository
git remote add origin https://github.com/ehxan139/neural-network-compression.git

# Push to GitHub
git push -u origin main
```

### Using SSH (If configured)

```bash
cd "C:\Users\ehsan\OneDrive\github_ehxan139\neural-network-compression"
git init
git add .
git commit -m "Initial commit: Neural network compression toolkit"
git branch -M main
git remote add origin git@github.com:ehxan139/neural-network-compression.git
git push -u origin main
```

## Verify Upload

After pushing, verify your repository at:
https://github.com/ehxan139/neural-network-compression

## Future Updates

To push future changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Create Release Tags

To create version releases:

```bash
# Create annotated tag
git tag -a v1.0.0 -m "Version 1.0.0 - Production-ready compression toolkit"

# Push tag
git push origin v1.0.0
```

## Troubleshooting

### If "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/ehxan139/neural-network-compression.git
```

### If push is rejected
```bash
# Pull remote changes first
git pull origin main --rebase

# Then push
git push origin main
```

---

**Note**: The `.gitignore` is configured to exclude:
- Model checkpoints (*.pt, *.pth)
- Training logs and TensorBoard events
- Virtual environments
- Large dataset files
- Benchmark results (can be regenerated)

Pre-trained models should be downloaded separately or shared via model registries (Hugging Face, Google Drive, etc.).
