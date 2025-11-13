#!/bin/bash
# Example run scripts for GPT-based data construction pipeline
# Choose the configuration that best fits your needs

set -e  # Exit on error

# Set your API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Set it with: export OPENAI_API_KEY='sk-your-key-here'"
    exit 1
fi

# Configuration
SOURCE_DATA="/data_ali/shunian/data/iceberg/scripts/data_clean.json"

# ============================================================================
# EXAMPLE 1: Quick Test (10 items, ~2 minutes, ~$1.30)
# ============================================================================
example_1_quick_test() {
    echo "Running Example 1: Quick Test (10 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/test_10 \
        --sample 10 \
        --examples-per-item 2 \
        --max-concurrent 5 \
        --batch-size 10

    echo "Test complete! Check ./gpt_output/test_10/"
}

# ============================================================================
# EXAMPLE 2: Pilot Run (100 items, ~45 minutes, ~$13)
# ============================================================================
example_2_pilot() {
    echo "Running Example 2: Pilot Run (100 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/pilot_100 \
        --sample 100 \
        --examples-per-item 2 \
        --max-concurrent 10 \
        --batch-size 50 \
        --checkpoint-interval 50

    echo "Pilot complete! Analyzing quality..."
    python gpt_pipeline_utils.py analyze-quality \
        --input ./gpt_output/pilot_100/generated_examples.jsonl \
        --sample 5
}

# ============================================================================
# EXAMPLE 3: Small Production (1K items, ~4 hours, ~$130)
# ============================================================================
example_3_small_production() {
    echo "Running Example 3: Small Production (1,000 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/production_1k \
        --sample 1000 \
        --examples-per-item 2 \
        --max-concurrent 10 \
        --batch-size 100 \
        --checkpoint-interval 200

    echo "Production run complete! Post-processing..."
    python data_quality_control.py \
        --input ./gpt_output/production_1k/generated_examples.jsonl \
        --output ./gpt_output/production_1k/final \
        --max-per-image 3
}

# ============================================================================
# EXAMPLE 4: Medium Production (10K items, ~30 hours, ~$1,300)
# ============================================================================
example_4_medium_production() {
    echo "Running Example 4: Medium Production (10,000 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/production_10k \
        --sample 10000 \
        --examples-per-item 2 \
        --max-concurrent 15 \
        --batch-size 200 \
        --checkpoint-interval 500
}

# ============================================================================
# EXAMPLE 5: Large Production (50K items, ~5 days, ~$6,500)
# ============================================================================
example_5_large_production() {
    echo "Running Example 5: Large Production (50,000 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/production_50k \
        --sample 50000 \
        --examples-per-item 2 \
        --max-concurrent 20 \
        --batch-size 200 \
        --checkpoint-interval 1000
}

# ============================================================================
# EXAMPLE 6: Full Scale (395K items, ~6 days, ~$51,000)
# ============================================================================
example_6_full_scale() {
    echo "Running Example 6: Full Scale (395,290 items)"
    echo "WARNING: This will cost approximately $51,000 and take 5-7 days"
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        return
    fi

    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/production_full \
        --examples-per-item 2 \
        --max-concurrent 20 \
        --batch-size 200 \
        --checkpoint-interval 2000
}

# ============================================================================
# EXAMPLE 7: Budget Mode - GPT-3.5 Only (10K items, ~$60)
# ============================================================================
example_7_budget_mode() {
    echo "Running Example 7: Budget Mode with GPT-3.5 (10,000 items)"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/budget_10k \
        --generation-model gpt-3.5-turbo \
        --validation-model gpt-3.5-turbo \
        --sample 10000 \
        --examples-per-item 2 \
        --max-concurrent 20 \
        --batch-size 200
}

# ============================================================================
# EXAMPLE 8: High Throughput (Tier 3+ API, 10K items, ~15 hours)
# ============================================================================
example_8_high_throughput() {
    echo "Running Example 8: High Throughput Mode (10,000 items)"
    echo "Note: Requires OpenAI Tier 3+ API access"
    python data_construction_gpt_pipeline.py \
        --api-key "$OPENAI_API_KEY" \
        --source "$SOURCE_DATA" \
        --output ./gpt_output/fast_10k \
        --sample 10000 \
        --examples-per-item 2 \
        --max-concurrent 40 \
        --batch-size 500 \
        --checkpoint-interval 1000
}

# ============================================================================
# EXAMPLE 9: Resume from Checkpoint
# ============================================================================
example_9_resume() {
    echo "Running Example 9: Resume from Checkpoint"
    echo "This will resume the last interrupted run"

    # Find the most recent output directory with a checkpoint
    CHECKPOINT_DIR=$(find ./gpt_output -name "checkpoint.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2 | xargs dirname)

    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "No checkpoint found in ./gpt_output/"
        return
    fi

    echo "Found checkpoint in: $CHECKPOINT_DIR"
    PROCESSED=$(jq .processed_items "$CHECKPOINT_DIR/checkpoint.json")
    echo "Resuming from item $PROCESSED"

    # Determine the original parameters (this is simplified - adjust as needed)
    # In practice, you'd save the command parameters in the checkpoint
    echo "Note: You should re-run the exact same command that was interrupted"
    echo "Example:"
    echo "python data_construction_gpt_pipeline.py --api-key \$OPENAI_API_KEY --source $SOURCE_DATA --output $CHECKPOINT_DIR --sample 10000 --examples-per-item 2 ..."
}

# ============================================================================
# EXAMPLE 10: Monitor Running Job
# ============================================================================
example_10_monitor() {
    echo "Running Example 10: Monitor Running Job"

    # Find the most recent output directory
    OUTPUT_DIR=$(find ./gpt_output -name "checkpoint.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2 | xargs dirname)

    if [ -z "$OUTPUT_DIR" ]; then
        echo "No active or recent jobs found"
        return
    fi

    echo "Monitoring: $OUTPUT_DIR"
    echo ""

    # Show checkpoint info
    python gpt_pipeline_utils.py inspect-checkpoint \
        --checkpoint "$OUTPUT_DIR/checkpoint.json"

    echo ""
    echo "To watch in real-time, run:"
    echo "  tail -f $OUTPUT_DIR/../gpt_pipeline.log"
    echo "  watch -n 10 'python gpt_pipeline_utils.py inspect-checkpoint --checkpoint $OUTPUT_DIR/checkpoint.json'"
}

# ============================================================================
# Main Menu
# ============================================================================
show_menu() {
    echo ""
    echo "========================================================================"
    echo "GPT Data Construction Pipeline - Example Runs"
    echo "========================================================================"
    echo ""
    echo "Select an example to run:"
    echo ""
    echo "  1) Quick Test          (10 items, ~2 min, ~\$1.30)"
    echo "  2) Pilot Run           (100 items, ~45 min, ~\$13)"
    echo "  3) Small Production    (1K items, ~4 hours, ~\$130)"
    echo "  4) Medium Production   (10K items, ~30 hours, ~\$1,300)"
    echo "  5) Large Production    (50K items, ~5 days, ~\$6,500)"
    echo "  6) Full Scale          (395K items, ~6 days, ~\$51,000)"
    echo "  7) Budget Mode         (10K items with GPT-3.5, ~\$60)"
    echo "  8) High Throughput     (10K items, fast, requires Tier 3+)"
    echo "  9) Resume from Checkpoint"
    echo " 10) Monitor Running Job"
    echo ""
    echo "  e) Estimate costs for custom configuration"
    echo "  q) Quit"
    echo ""
    echo "========================================================================"
}

# Cost estimation helper
estimate_costs() {
    echo ""
    read -p "Number of items to process: " num_items
    read -p "Examples per item (default 2): " examples_per_item
    examples_per_item=${examples_per_item:-2}

    read -p "Generation model (gpt-4-turbo-preview/gpt-3.5-turbo): " gen_model
    gen_model=${gen_model:-gpt-4-turbo-preview}

    read -p "Validation model (gpt-3.5-turbo): " val_model
    val_model=${val_model:-gpt-3.5-turbo}

    python gpt_pipeline_utils.py estimate-cost \
        --items "$num_items" \
        --examples-per-item "$examples_per_item" \
        --generation-model "$gen_model" \
        --validation-model "$val_model"
}

# Main loop
main() {
    while true; do
        show_menu
        read -p "Enter your choice: " choice

        case $choice in
            1) example_1_quick_test ;;
            2) example_2_pilot ;;
            3) example_3_small_production ;;
            4) example_4_medium_production ;;
            5) example_5_large_production ;;
            6) example_6_full_scale ;;
            7) example_7_budget_mode ;;
            8) example_8_high_throughput ;;
            9) example_9_resume ;;
            10) example_10_monitor ;;
            e|E) estimate_costs ;;
            q|Q) echo "Goodbye!"; exit 0 ;;
            *) echo "Invalid choice. Please try again." ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main menu if script is executed directly
if [ "${BASH_SOURCE[0]}" -eq "${0}" ]; then
    main
fi

# Individual functions can also be called directly:
# ./run_gpt_pipeline_examples.sh example_1_quick_test
