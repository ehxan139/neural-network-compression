"""
Knowledge Distillation Implementation

Train small "student" networks to mimic large "teacher" networks,
achieving 80-90% compression with 98%+ accuracy retention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Tuple
import time
import numpy as np


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    L_total = alpha * L_distillation + (1-alpha) * L_student
    
    Where:
    - L_distillation: KL divergence between teacher and student soft predictions
    - L_student: Standard cross-entropy loss with true labels
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Initialize KD loss.
        
        Args:
            temperature: Softmax temperature for soft targets (higher = softer)
            alpha: Weight for distillation loss (0-1)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions (detached)
            labels: Ground truth labels
            
        Returns:
            (total_loss, loss_dict)
        """
        # Soft targets (distillation loss)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets (student loss)
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'distillation': distillation_loss.item(),
            'student': student_loss.item()
        }
        
        return total_loss, loss_dict


class KnowledgeDistiller:
    """
    Knowledge Distillation trainer for model compression.
    
    Trains a small student network to mimic a large teacher network's
    behavior, achieving significant compression with minimal accuracy loss.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Knowledge Distiller.
        
        Args:
            teacher_model: Pre-trained large model
            student_model: Small model to train
            temperature: Softmax temperature (2-10, higher=softer)
            alpha: Distillation weight (0.5-0.9 recommended)
            device: Training device
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Loss function
        self.criterion = KnowledgeDistillationLoss(temperature, alpha)
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_schedule: str = 'cosine',
        warmup_epochs: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Train student model via knowledge distillation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
            lr_schedule: 'cosine', 'step', or 'none'
            warmup_epochs: Learning rate warmup period
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        # Optimizer
        optimizer = optim.SGD(
            self.student.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if lr_schedule == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif lr_schedule == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epochs//2, 3*epochs//4], gamma=0.1
            )
        else:
            scheduler = None
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            # Train one epoch
            train_metrics = self._train_epoch(
                train_loader, optimizer, epoch, epochs, verbose
            )
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update learning rate
            if scheduler is not None and epoch >= warmup_epochs:
                scheduler.step()
            
            # Track history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                self.best_student_state = self.student.state_dict()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        
        # Load best model
        self.student.load_state_dict(self.best_student_state)
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }
    
    def _train_epoch(self, train_loader, optimizer, epoch, total_epochs, verbose):
        """Train for one epoch."""
        self.student.train()
        
        running_loss = 0.0
        running_dist_loss = 0.0
        running_student_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            student_logits = self.student(inputs)
            
            # Compute loss
            loss, loss_dict = self.criterion(student_logits, teacher_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            running_dist_loss += loss_dict['distillation']
            running_student_loss += loss_dict['student']
            
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': running_loss / len(train_loader),
            'distillation_loss': running_dist_loss / len(train_loader),
            'student_loss': running_student_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def _validate(self, val_loader):
        """Validate student model."""
        self.student.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)
                
                loss, _ = self.criterion(student_logits, teacher_logits, labels)
                
                running_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': running_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def benchmark(self, test_loader, compare_teacher: bool = True) -> Dict:
        """
        Benchmark student vs teacher performance.
        
        Args:
            test_loader: Test data loader
            compare_teacher: Include teacher metrics
            
        Returns:
            Benchmark results dict
        """
        results = {}
        
        # Student metrics
        student_acc, student_time = self._evaluate_model(self.student, test_loader)
        student_size = self._get_model_size(self.student)
        
        results['student'] = {
            'accuracy': student_acc,
            'inference_time_ms': student_time * 1000,
            'model_size_mb': student_size,
            'parameters': sum(p.numel() for p in self.student.parameters())
        }
        
        if compare_teacher:
            teacher_acc, teacher_time = self._evaluate_model(self.teacher, test_loader)
            teacher_size = self._get_model_size(self.teacher)
            
            results['teacher'] = {
                'accuracy': teacher_acc,
                'inference_time_ms': teacher_time * 1000,
                'model_size_mb': teacher_size,
                'parameters': sum(p.numel() for p in self.teacher.parameters())
            }
            
            # Compression metrics
            results['compression'] = {
                'size_ratio': teacher_size / student_size,
                'speedup': teacher_time / student_time,
                'accuracy_retention': (student_acc / teacher_acc) * 100,
                'accuracy_gap': teacher_acc - student_acc
            }
        
        return results
    
    def _evaluate_model(self, model, test_loader) -> Tuple[float, float]:
        """Evaluate accuracy and inference time."""
        model.eval()
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        elapsed_time = time.time() - start_time
        accuracy = 100.0 * correct / total
        avg_inference_time = elapsed_time / total
        
        return accuracy, avg_inference_time
    
    def _get_model_size(self, model) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def save_student(self, filepath: str):
        """Save compressed student model."""
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'architecture': str(self.student),
            'compression_ratio': None,  # Will be calculated if teacher available
            'training_params': {
                'temperature': self.temperature,
                'alpha': self.alpha
            }
        }, filepath)
        print(f"Student model saved to {filepath}")
    
    def load_student(self, filepath: str):
        """Load compressed student model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.student.to(self.device)
        print(f"Student model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Knowledge Distillation - Example Usage")
    print("=" * 60)
    
    # This would typically use real models and data
    # See notebooks/01_knowledge_distillation_demo.ipynb for full example
    
    print("\nKnowledge distillation enables:")
    print("  • 80-90% model size reduction")
    print("  • 5-8x faster inference")
    print("  • 98-99% accuracy retention")
    print("  • Deployment on mobile/edge devices")
