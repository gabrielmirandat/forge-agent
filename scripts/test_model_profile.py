"""Test script to investigate when model.profile is available in ChatOllama.

This script tests different scenarios to understand when and how to access
model capabilities via the profile attribute.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_profile_access():
    """Test accessing model profile at different stages."""
    
    print("=" * 80)
    print("TESTING MODEL PROFILE ACCESS")
    print("=" * 80)
    print()
    
    try:
        # Try to import ChatOllama
        try:
            from langchain_ollama import ChatOllama
            print("✅ Using langchain_ollama")
            package_name = "langchain_ollama"
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                print("⚠️ Using langchain_community (deprecated)")
                package_name = "langchain_community"
            except ImportError:
                print("❌ Could not import ChatOllama")
                return
        
        # Check versions
        print("\n" + "=" * 80)
        print("VERSION INFORMATION")
        print("=" * 80)
        
        try:
            import langchain_core
            print(f"langchain-core: {langchain_core.__version__}")
        except:
            print("Could not get langchain-core version")
        
        try:
            if package_name == "langchain_ollama":
                import langchain_ollama
                print(f"langchain-ollama: {langchain_ollama.__version__}")
        except:
            pass
        
        # Test 1: Create instance and check profile immediately
        print("\n" + "=" * 80)
        print("TEST 1: Profile immediately after creation")
        print("=" * 80)
        
        model1 = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.0,
            num_predict=4096,
        )
        
        print(f"Model created: {model1}")
        print(f"Model type: {type(model1)}")
        print(f"Has profile attribute: {hasattr(model1, 'profile')}")
        
        if hasattr(model1, 'profile'):
            try:
                profile = model1.profile
                print(f"Profile value: {profile}")
                print(f"Profile type: {type(profile)}")
                if profile is not None:
                    if isinstance(profile, dict):
                        print(f"Profile keys: {list(profile.keys())}")
                        print(f"Profile content: {profile}")
                    else:
                        print(f"Profile is not a dict: {profile}")
                        # Try to convert to dict
                        if hasattr(profile, '__dict__'):
                            print(f"Profile __dict__: {profile.__dict__}")
                        if hasattr(profile, 'model_dump'):
                            print(f"Profile model_dump: {profile.model_dump()}")
            except Exception as e:
                print(f"❌ Error accessing profile: {e}")
        else:
            print("❌ No profile attribute found")
        
        # Test 2: Check after accessing model property
        print("\n" + "=" * 80)
        print("TEST 2: Profile after accessing .model property")
        print("=" * 80)
        
        if hasattr(model1, 'model'):
            print(f"Has .model property: {hasattr(model1, 'model')}")
            try:
                model_obj = model1.model
                print(f"Model object: {model_obj}")
                print(f"Model object type: {type(model_obj)}")
                print(f"Model object has profile: {hasattr(model_obj, 'profile')}")
                if hasattr(model_obj, 'profile'):
                    try:
                        profile = model_obj.profile
                        print(f"Model object profile: {profile}")
                        if profile:
                            print(f"Profile type: {type(profile)}")
                    except Exception as e:
                        print(f"❌ Error accessing model.profile: {e}")
            except Exception as e:
                print(f"❌ Error accessing .model: {e}")
        
        # Test 3: Check profile on different models
        print("\n" + "=" * 80)
        print("TEST 3: Profile on different models")
        print("=" * 80)
        
        models_to_test = ["qwen3:8b", "mistral:latest", "deepseek-r1:8b"]
        
        for model_name in models_to_test:
            print(f"\n--- Testing {model_name} ---")
            try:
                model = ChatOllama(
                    model=model_name,
                    base_url="http://localhost:11434",
                    temperature=0.0,
                )
                print(f"✅ Created: {model_name}")
                print(f"Has profile: {hasattr(model, 'profile')}")
                
                if hasattr(model, 'profile'):
                    try:
                        profile = model.profile
                        print(f"Profile: {profile}")
                        if profile:
                            if isinstance(profile, dict):
                                print(f"Profile keys: {list(profile.keys())}")
                            elif hasattr(profile, 'model_dump'):
                                print(f"Profile (model_dump): {profile.model_dump()}")
                            elif hasattr(profile, '__dict__'):
                                print(f"Profile (__dict__): {profile.__dict__}")
                            else:
                                print(f"Profile type: {type(profile)}")
                        else:
                            print("⚠️ Profile is None")
                    except Exception as e:
                        print(f"❌ Error accessing profile: {e}")
                else:
                    print("❌ No profile attribute")
            except Exception as e:
                print(f"❌ Error creating model: {e}")
        
        # Test 4: Check if profile needs to be loaded explicitly
        print("\n" + "=" * 80)
        print("TEST 4: Checking profile loading mechanism")
        print("=" * 80)
        
        model4 = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
        )
        
        # Check model's __dict__ and attributes
        print(f"Model __dict__ keys: {list(model4.__dict__.keys())}")
        print(f"Model dir() with 'profile': {[m for m in dir(model4) if 'profile' in m.lower()]}")
        
        # Check if profile is a property
        import inspect
        try:
            profile_descriptor = inspect.getattr_static(type(model4), 'profile', None)
            print(f"Profile descriptor: {profile_descriptor}")
            print(f"Is property: {isinstance(profile_descriptor, property)}")
        except:
            pass
        
        # Test 5: Try to access profile after invoke (if possible)
        print("\n" + "=" * 80)
        print("TEST 5: Profile after invoke (if Ollama is available)")
        print("=" * 80)
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    print("✅ Ollama is available, testing invoke...")
                    try:
                        from langchain_core.messages import HumanMessage
                        result = await model4.ainvoke([HumanMessage(content="test")])
                        print(f"✅ Invoke successful")
                        print(f"Has profile after invoke: {hasattr(model4, 'profile')}")
                        if hasattr(model4, 'profile'):
                            profile = model4.profile
                            print(f"Profile after invoke: {profile}")
                    except Exception as e:
                        print(f"⚠️ Invoke failed (expected if model not loaded): {e}")
                else:
                    print("⚠️ Ollama not responding, skipping invoke test")
        except:
            print("⚠️ Could not connect to Ollama, skipping invoke test")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_profile_access())
