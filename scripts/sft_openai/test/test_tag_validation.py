#!/usr/bin/env python3
"""
测试tag结构验证功能

测试各种合法和非法的tag结构
"""

import sys
sys.path.insert(0, '/data_ali/shunian/verl/scripts/sft_openai')

from data_construction_gpt_pipeline import GPTDataGenerator


def test_tag_validation():
    """测试tag结构验证"""

    # 创建一个generator实例（只用于测试validation方法）
    # 这里API key可以是假的，因为我们只测试validate_tag_structure方法
    generator = GPTDataGenerator(
        api_key="test",
        base_url="test",
        generation_model="test",
        validation_model="test"
    )

    print("=" * 80)
    print("Tag Structure Validation Tests")
    print("=" * 80)

    # Test cases
    test_cases = [
        # Valid cases
        {
            "name": "Valid: look-think-answer",
            "response": "<look>Observing the image</look><think>Analyzing the content</think><answer>Final answer</answer>",
            "should_pass": True
        },
        {
            "name": "Valid: think-look-answer",
            "response": "<think>Initial thought</think><look>Looking at details</look><answer>Conclusion</answer>",
            "should_pass": True
        },
        {
            "name": "Valid: multiple cycles",
            "response": "<look>First look</look><think>First think</think><look>Second look</look><think>Second think</think><answer>Answer</answer>",
            "should_pass": True
        },
        {
            "name": "Valid: three cycles",
            "response": "<look>L1</look><think>T1</think><look>L2</look><think>T2</think><look>L3</look><think>T3</think><answer>A</answer>",
            "should_pass": True
        },

        # Invalid cases - missing answer
        {
            "name": "Invalid: no answer tag",
            "response": "<look>Looking</look><think>Thinking</think>",
            "should_pass": False,
            "expected_error": "no_final_answer"
        },
        {
            "name": "Invalid: answer not at end",
            "response": "<look>L</look><answer>A</answer><think>T</think>",
            "should_pass": False,
            "expected_error": "no_final_answer"
        },

        # Invalid cases - not alternating
        {
            "name": "Invalid: two looks in a row",
            "response": "<look>L1</look><look>L2</look><answer>A</answer>",
            "should_pass": False,
            "expected_error": "not_alternating"
        },
        {
            "name": "Invalid: two thinks in a row",
            "response": "<think>T1</think><think>T2</think><answer>A</answer>",
            "should_pass": False,
            "expected_error": "not_alternating"
        },
        {
            "name": "Invalid: look-look-think",
            "response": "<look>L1</look><look>L2</look><think>T</think><answer>A</answer>",
            "should_pass": False,
            "expected_error": "not_alternating"
        },

        # Invalid cases - text outside tags
        {
            "name": "Invalid: text before tags",
            "response": "Some intro text <look>L</look><think>T</think><answer>A</answer>",
            "should_pass": False,
            "expected_error": "text_before_tags"
        },
        {
            "name": "Invalid: text between tags",
            "response": "<look>L</look> extra text <think>T</think><answer>A</answer>",
            "should_pass": False,
            "expected_error": "text_between_tags"
        },
        {
            "name": "Invalid: text after tags",
            "response": "<look>L</look><think>T</think><answer>A</answer> extra text",
            "should_pass": False,
            "expected_error": "text_after_tags"
        },

        # Invalid cases - empty tags
        {
            "name": "Invalid: empty look tag",
            "response": "<look></look><think>T</think><answer>A</answer>",
            "should_pass": False,
            "expected_error": "empty_tag_content"
        },
        {
            "name": "Invalid: empty answer tag",
            "response": "<look>L</look><think>T</think><answer></answer>",
            "should_pass": False,
            "expected_error": "empty_tag_content"
        },

        # Invalid cases - multiple answers
        {
            "name": "Invalid: multiple answer tags",
            "response": "<look>L</look><answer>A1</answer><answer>A2</answer>",
            "should_pass": False,
            "expected_error": "multiple_answers"
        },

        # Invalid cases - no look/think before answer
        {
            "name": "Invalid: only answer tag",
            "response": "<answer>Just an answer</answer>",
            "should_pass": False,
            "expected_error": "no_look_think"
        },

        # Edge cases with whitespace
        {
            "name": "Valid: whitespace around tags (should be ok)",
            "response": "  <look>L</look>  \n  <think>T</think>  \n  <answer>A</answer>  ",
            "should_pass": True
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        is_valid, result = generator.validate_tag_structure(test["response"])

        # Check if result matches expectation
        if is_valid == test["should_pass"]:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        print(f"\n{i}. {test['name']}")
        print(f"   Expected: {'Valid' if test['should_pass'] else 'Invalid'}")
        print(f"   Got: {'Valid' if is_valid else 'Invalid'}")
        print(f"   Status: {status}")

        if not is_valid:
            error = result.get('error', 'unknown')
            message = result.get('message', 'No message')
            print(f"   Error: {error}")
            print(f"   Message: {message}")

            # Check if error type matches expected
            if 'expected_error' in test and error != test['expected_error']:
                print(f"   ⚠️  Expected error '{test['expected_error']}', got '{error}'")
        else:
            tag_seq = result.get('tag_sequence', [])
            print(f"   Tag sequence: {tag_seq}")

    print("\n" + "=" * 80)
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = test_tag_validation()
    sys.exit(0 if success else 1)
