#!/bin/bash
# Quick Start Script for IMDB Federated Learning Project
# This script demonstrates common workflows

set -e  # Exit on error

echo "=================================="
echo "IMDB FL Quick Start"
echo "=================================="
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "fl_imdb" ]; then
    echo "⚠️  Warning: fl_imdb conda environment not activated"
    echo "   Run: conda activate fl_imdb"
    echo ""
fi

# Function to display menu
show_menu() {
    echo "Select an option:"
    echo "1) Run centralized training"
    echo "2) Run federated learning training"
    echo "3) Run evaluation (requires trained models)"
    echo "4) Run complete pipeline (all three)"
    echo "5) Clean outputs"
    echo "6) Exit"
    echo ""
}

# Function to run centralized training
run_centralized() {
    echo ""
    echo "=================================="
    echo "Running Centralized Training"
    echo "=================================="
    python src/training/centralized.py
    echo ""
    echo "✅ Centralized training complete!"
    echo "   Model saved to: outputs/models/centralized.pt"
    echo "   Metrics saved to: outputs/logs/centralized_metrics.json"
    echo ""
}

# Function to run federated training
run_federated() {
    echo ""
    echo "=================================="
    echo "Running Federated Learning"
    echo "=================================="
    python src/training/federated.py
    echo ""
    echo "✅ Federated training complete!"
    echo "   Model saved to: outputs/models/federated.pt"
    echo "   Metrics saved to: outputs/logs/federated_metrics.json"
    echo ""
}

# Function to run evaluation
run_evaluation() {
    echo ""
    echo "=================================="
    echo "Running Evaluation"
    echo "=================================="
    
    # Check if models exist
    if [ ! -f "outputs/models/centralized.pt" ] || [ ! -f "outputs/models/federated.pt" ]; then
        echo "❌ Error: Trained models not found!"
        echo "   Please run training first (options 1 and 2)"
        return 1
    fi
    
    python src/evaluation/evaluate.py
    echo ""
    echo "✅ Evaluation complete!"
    echo "   Plots saved to: outputs/plots/"
    echo ""
}

# Function to clean outputs
clean_outputs() {
    echo ""
    echo "Cleaning outputs directory..."
    if [ -d "outputs" ]; then
        rm -rf outputs/models/* outputs/logs/* outputs/plots/*
        echo "✅ Outputs cleaned!"
    else
        echo "ℹ️  No outputs directory found"
    fi
    echo ""
}

# Function to run complete pipeline
run_complete() {
    echo ""
    echo "=================================="
    echo "Running Complete Pipeline"
    echo "=================================="
    echo "This will take approximately 30-60 minutes"
    echo ""
    
    run_centralized
    run_federated
    run_evaluation
    
    echo ""
    echo "=================================="
    echo "🎉 Complete Pipeline Finished!"
    echo "=================================="
    echo "Check the following locations:"
    echo "  - Models: outputs/models/"
    echo "  - Metrics: outputs/logs/"
    echo "  - Plots: outputs/plots/"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [1-6]: " choice
    
    case $choice in
        1)
            run_centralized
            ;;
        2)
            run_federated
            ;;
        3)
            run_evaluation
            ;;
        4)
            run_complete
            ;;
        5)
            clean_outputs
            ;;
        6)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please select 1-6."
            echo ""
            ;;
    esac
done
