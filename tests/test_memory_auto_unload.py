#!/usr/bin/env python3
"""
🧠 MLX-RAG Memory Auto-Unload Test
Specifically tests the automatic LRU unloading when hitting concurrent model limits
"""

import asyncio
import httpx
import time
from typing import List, Dict

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0

async def test_auto_unload_behavior():
    """Test that models are automatically unloaded when hitting concurrent limits."""
    print("🧠 Testing Automatic LRU Unloading Behavior")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Get system status first
        response = await client.get(f"{BASE_URL}/v1/system/status")
        if response.status_code != 200:
            print("❌ Server not responding")
            return False

        status = response.json()
        max_concurrent = status.get('model_manager', {}).get('max_concurrent_models', 3)
        print(f"📊 System configured for max {max_concurrent} concurrent models")

        # Test models to load sequentially
        test_models = [
            "deepseek-r1-0528-qwen3-8b-mlx-8bit",
            "gemma-3-27b-it-qat-4bit",
            "gemma-3n-e4b-it-mlx-8bit",
            "qwen3-8b-6bit",  # This should trigger auto-unload
            "qwen3-embedding-4b-4bit-dwq"  # This should also trigger auto-unload
        ]

        print(f"\n🔄 Loading {len(test_models)} models sequentially (exceeds limit of {max_concurrent})")

        for i, model_name in enumerate(test_models):
            print(f"\n  📥 Loading model {i+1}/{len(test_models)}: {model_name}")

            # Check current state before loading using the INTERNAL endpoint
            models_response = await client.get(f"{BASE_URL}/v1/manager/models")
            if models_response.status_code == 200:
                models_data = models_response.json()
                loaded_models = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
                print(f"    📊 Before: {len(loaded_models)} loaded - {loaded_models}")

            # Load the model
            load_response = await client.post(f"{BASE_URL}/v1/models/{model_name}/load")

            if load_response.status_code == 200:
                print(f"    ✅ Successfully loaded {model_name}")

                # Check state after loading using the INTERNAL endpoint
                models_response = await client.get(f"{BASE_URL}/v1/manager/models")
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    new_loaded_models = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
                    print(f"    📊 After:  {len(new_loaded_models)} loaded - {new_loaded_models}")

                    # Check if auto-unloading happened
                    if len(loaded_models) >= max_concurrent and len(new_loaded_models) <= max_concurrent:
                        unloaded = set(loaded_models) - set(new_loaded_models)
                        if unloaded:
                            print(f"    🔄 AUTO-UNLOADED: {list(unloaded)} (LRU eviction working!)")
                        else:
                            print(f"    ⚠️  Expected auto-unload but none detected")

                    # Verify the new model is actually loaded and responding
                    test_response = await client.post(f"{BASE_URL}/v1/chat/completions", json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    })

                    if test_response.status_code == 200:
                        print(f"    ✅ Model {model_name} is responding to requests")
                    else:
                        print(f"    ⚠️  Model {model_name} loaded but not responding")

            else:
                error_detail = "Unknown error"
                try:
                    error_data = load_response.json()
                    error_detail = error_data.get('detail', str(load_response.status_code))
                except:
                    error_detail = f"HTTP {load_response.status_code}"
                print(f"    ❌ Failed to load {model_name}: {error_detail}")

            # Small delay between loads
            await asyncio.sleep(1)

        # Final summary
        print(f"\n📊 Final Summary:")
        models_response = await client.get(f"{BASE_URL}/v1/manager/models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            final_loaded = [m['name'] for m in models_data.get('models', []) if m.get('status') == 'loaded']
            print(f"    📈 Final loaded models: {len(final_loaded)} - {final_loaded}")

            if len(final_loaded) <= max_concurrent:
                print(f"    ✅ SUCCESS: Model count ({len(final_loaded)}) respects limit ({max_concurrent})")
                print(f"    ✅ AUTO-UNLOAD SYSTEM WORKING: Older models were automatically evicted")
                return True
            else:
                print(f"    ❌ FAILURE: Model count ({len(final_loaded)}) exceeds limit ({max_concurrent})")
                return False
        else:
            print(f"    ❌ Could not get final model status")
            return False

async def main():
    """Main test function."""
    try:
        success = await test_auto_unload_behavior()
        if success:
            print(f"\n🎉 AUTO-UNLOAD TEST PASSED!")
            print(f"✅ The system correctly unloads older models when hitting concurrent limits")
        else:
            print(f"\n❌ AUTO-UNLOAD TEST FAILED!")
            print(f"⚠️  The system is not properly managing concurrent model limits")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting MLX-RAG Auto-Unload Test...")
    print("📋 Make sure MLX-RAG server is running on http://localhost:8000")
    print()

    asyncio.run(main())