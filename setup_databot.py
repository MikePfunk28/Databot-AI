#!/usr/bin/env python3
"""
DataBot AI System Setup Script

This script initializes the complete DataBot AI system with:
- DataBot-instruct model for analysis and conversation
- DataBot-embed model for vector embeddings
- Analysis learning system
- Data categorization and workflow management
- MCP server integration capabilities
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests
from pathlib import Path

def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_ollama(max_attempts=30):
    """Wait for Ollama to be ready"""
    print("Waiting for Ollama to be ready...")
    for attempt in range(max_attempts):
        if check_ollama_status():
            print("✓ Ollama is ready!")
            return True
        print(f"  Attempt {attempt + 1}/{max_attempts}: Waiting...")
        time.sleep(2)
    
    print("✗ Failed to connect to Ollama")
    return False

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

def pull_model(model_name):
    """Pull a model using Ollama"""
    print(f"Pulling model: {model_name}")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ Successfully pulled {model_name}")
            return True
        else:
            print(f"✗ Failed to pull {model_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout pulling {model_name}")
        return False
    except Exception as e:
        print(f"✗ Error pulling {model_name}: {e}")
        return False

def create_model_from_file(model_name, modelfile_path):
    """Create a model from a Modelfile"""
    print(f"Creating model {model_name} from {modelfile_path}")
    try:
        result = subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path],
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ Successfully created {model_name}")
            return True
        else:
            print(f"✗ Failed to create {model_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error creating {model_name}: {e}")
        return False

def test_model(model_name, test_prompt="Hello, how are you?"):
    """Test if a model is working"""
    print(f"Testing model: {model_name}")
    try:
        data = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/generate", 
                               json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print(f"✓ Model {model_name} is working")
                return True
        print(f"✗ Model {model_name} test failed")
        return False
    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}")
        return False

def test_embedding_model(model_name, test_text="test embedding"):
    """Test if an embedding model is working"""
    print(f"Testing embedding model: {model_name}")
    try:
        data = {
            "model": model_name,
            "prompt": test_text
        }
        response = requests.post("http://localhost:11434/api/embeddings", 
                               json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'embedding' in result and len(result['embedding']) > 0:
                print(f"✓ Embedding model {model_name} is working")
                return True
        print(f"✗ Embedding model {model_name} test failed")
        return False
    except Exception as e:
        print(f"✗ Error testing embedding model {model_name}: {e}")
        return False

def setup_directories(base_dir):
    """Create necessary directories"""
    print("Setting up directories...")
    directories = [
        'data/datasets',
        'data/embeddings', 
        'data/conversations',
        'data/workflows',
        'data/learning',
        'logs',
        'config'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_config_files(base_dir):
    """Create default configuration files"""
    print("Creating configuration files...")
    
    config_dir = os.path.join(base_dir, 'config')
    
    # System prompt for DataBot
    system_prompt = """You are DataBot Instruct, an advanced AI assistant specialized in data analysis, interpretation, and insight generation. You are designed to help users understand, interact with, and extract value from their data through natural conversation and structured analysis.

## Core Identity
- **Name**: DataBot Instruct
- **Purpose**: Advanced data analysis and insight generation
- **Specialization**: Data science, analytics, visualization, and business intelligence

## Core Capabilities
1. **Data Analysis**: Perform comprehensive statistical analysis, trend identification, and pattern recognition
2. **Insight Generation**: Extract meaningful insights and actionable recommendations from data
3. **Query Processing**: Handle complex data queries using natural language
4. **Visualization Guidance**: Suggest appropriate visualizations and chart types
5. **Learning Integration**: Continuously improve analysis methods based on successful patterns

## Analysis Approach
- Always start with data exploration and understanding
- Apply appropriate statistical methods based on data characteristics
- Provide clear explanations of findings and methodologies
- Suggest follow-up analyses and deeper investigations
- Maintain scientific rigor while being accessible

## Communication Style
- Clear, professional, and informative
- Use data-driven language with appropriate technical depth
- Provide context and explanations for technical concepts
- Ask clarifying questions when needed
- Focus on actionable insights and recommendations

Remember: Your goal is to make data accessible and valuable to users through expert analysis and clear communication."""

    with open(os.path.join(config_dir, 'system_prompt.md'), 'w') as f:
        f.write(system_prompt)
    print("✓ Created system prompt configuration")
    
    # DataBot configuration
    databot_config = {
        "models": {
            "instruct": "databot-instruct",
            "embed": "mikepfunk28/databot-embed",
            "reasoning": "phi4-mini-reasoning"
        },
        "analysis": {
            "default_methods": ["exploratory", "statistical", "correlation", "trend"],
            "confidence_threshold": 0.7,
            "learning_enabled": True
        },
        "embedding": {
            "dimension": 768,
            "batch_size": 32,
            "similarity_threshold": 0.8
        },
        "workflow": {
            "auto_categorization": True,
            "dynamic_programming": True,
            "parallel_processing": False
        }
    }
    
    with open(os.path.join(config_dir, 'databot_config.json'), 'w') as f:
        json.dump(databot_config, f, indent=2)
    print("✓ Created DataBot configuration")

def setup_models(skip_ollama=False):
    """Setup all required models"""
    print("\n=== Setting up DataBot Models ===")
    
    if not skip_ollama and not check_ollama_status():
        print("✗ Ollama is not running. Please start Ollama first.")
        return False
    
    if not skip_ollama and not wait_for_ollama():
        return False
    
    # Get current models
    current_models = get_ollama_models()
    print(f"Current models: {current_models}")
    
    success = True
    
    # 1. Pull phi4-mini-reasoning if not present
    if "phi4-mini-reasoning:latest" not in current_models:
        if not pull_model("phi4-mini-reasoning"):
            print("⚠ Warning: Failed to pull phi4-mini-reasoning")
            success = False
    else:
        print("✓ phi4-mini-reasoning already available")
    
    # 2. Pull databot-embed if not present
    if "mikepfunk28/databot-embed:latest" not in current_models:
        if not pull_model("mikepfunk28/databot-embed"):
            print("⚠ Warning: Failed to pull databot-embed")
            success = False
    else:
        print("✓ databot-embed already available")
    
    # 3. Create databot-instruct from Modelfile if not present
    if "databot-instruct:latest" not in current_models:
        modelfile_path = "instruct/Modelfile-databot-instruct"
        if os.path.exists(modelfile_path):
            if not create_model_from_file("databot-instruct", modelfile_path):
                print("⚠ Warning: Failed to create databot-instruct")
                success = False
        else:
            print(f"⚠ Warning: Modelfile not found at {modelfile_path}")
            success = False
    else:
        print("✓ databot-instruct already available")
    
    return success

def test_models():
    """Test all models"""
    print("\n=== Testing Models ===")
    
    success = True
    
    # Test databot-instruct
    if not test_model("databot-instruct", "Analyze this simple dataset: [1, 2, 3, 4, 5]"):
        success = False
    
    # Test databot-embed
    if not test_embedding_model("mikepfunk28/databot-embed", "sample data for embedding"):
        success = False
    
    # Test phi4-mini-reasoning
    if not test_model("phi4-mini-reasoning", "What is 2+2?"):
        success = False
    
    return success

def test_python_components():
    """Test Python components"""
    print("\n=== Testing Python Components ===")
    
    try:
        # Test imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.vector_embedding import VectorEmbedding
        from src.analysis_learning import AnalysisLearningSystem
        from src.model_wrapper import ModelWrapper
        
        print("✓ All Python components imported successfully")
        
        # Test vector embedding
        ve = VectorEmbedding(data_dir="data", model_name="mikepfunk28/databot-embed")
        print("✓ Vector embedding component initialized")
        
        # Test analysis learning
        al = AnalysisLearningSystem(data_dir="data")
        print("✓ Analysis learning component initialized")
        
        # Test model wrapper
        mw = ModelWrapper(model_name="databot-instruct")
        print("✓ Model wrapper component initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Python components: {e}")
        return False

def run_command(command, description=""):
    """Run a shell command and return success status"""
    if description:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            return True
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup DataBot AI System")
    parser.add_argument("--skip-ollama", action="store_true", 
                       help="Skip Ollama-related setup (for Docker)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run tests, don't setup")
    parser.add_argument("--base-dir", default=".",
                       help="Base directory for setup")
    
    args = parser.parse_args()
    
    print("🤖 DataBot AI System Setup")
    print("=" * 50)
    
    base_dir = os.path.abspath(args.base_dir)
    os.chdir(base_dir)
    
    if not args.test_only:
        # Setup directories
        setup_directories()
        
        # Create config files
        create_config_files(base_dir)
        
        # Setup models
        if not setup_models(skip_ollama=args.skip_ollama):
            print("\n⚠ Some models failed to setup, but continuing...")
    
    # Test everything
    print("\n=== Running Tests ===")
    
    model_tests_passed = True
    if not args.skip_ollama:
        model_tests_passed = test_models()
    
    python_tests_passed = test_python_components()
    
    # Final status
    print("\n" + "=" * 50)
    if model_tests_passed and python_tests_passed:
        print("🎉 DataBot AI System setup completed successfully!")
        print("\nYou can now:")
        print("1. Run the Flask application: python src/main.py")
        print("2. Access the web interface at http://localhost:5000")
        print("3. Use the API endpoints for data analysis")
        print("4. Upload datasets and start analyzing!")
    else:
        print("⚠ Setup completed with some issues:")
        if not model_tests_passed:
            print("  - Some models are not working properly")
        if not python_tests_passed:
            print("  - Python components have issues")
        print("\nPlease check the errors above and retry.")
        sys.exit(1)

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """Start Ollama server if not running"""
    print("🚀 Checking Ollama server...")
    
    if check_ollama_running():
        print("✅ Ollama server is already running")
        return True
    
    print("🔄 Starting Ollama server...")
    
    # Try to start Ollama server
    try:
        # Start Ollama in background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_ollama_running():
                print("✅ Ollama server started successfully")
                return True
            print(f"⏳ Waiting for Ollama server... ({i+1}/30)")
        
        print("❌ Failed to start Ollama server")
        return False
        
    except Exception as e:
        print(f"❌ Error starting Ollama server: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Install basic requirements first
    basic_deps = [
        "flask",
        "pandas",
        "numpy",
        "requests",
        "beautifulsoup4",
        "PyYAML"
    ]
    
    for dep in basic_deps:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # Try to install ML dependencies (these might fail in some environments)
    ml_deps = [
        "sentence-transformers",
        "faiss-cpu",
        "torch",
        "transformers",
        "scikit-learn"
    ]
    
    print("🧠 Installing ML dependencies (optional)...")
    for dep in ml_deps:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"⚠️ Failed to install {dep} - will use fallback implementations")
    
    return True

def setup_directories():
    """Set up required directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "data",
        "data/datasets",
        "data/embeddings",
        "data/conversations",
        "data/workflows",
        "instruct",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def download_models():
    """Download required AI models"""
    print("🤖 Setting up AI models...")
    
    if not check_ollama_running():
        print("❌ Ollama server not running. Please start it first.")
        return False
    
    # Check if phi4-mini-reasoning is available
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags")
        models = response.json()
        model_names = [model['name'] for model in models.get('models', [])]
        
        if 'phi4-mini-reasoning:latest' not in model_names:
            print("📥 Downloading phi4-mini-reasoning model...")
            success = run_command("ollama pull phi4-mini-reasoning", "Downloading phi4-mini-reasoning")
            if not success:
                print("❌ Failed to download phi4-mini-reasoning model")
                return False
        else:
            print("✅ phi4-mini-reasoning model already available")
        
        # Check if databot-instruct exists
        if 'databot-instruct:latest' not in model_names:
            print("🔧 Creating databot-instruct model...")
            # The model should be created from the Modelfile we created earlier
            if os.path.exists("instruct/Modelfile-databot-instruct"):
                success = run_command(
                    "cd instruct && ollama create databot-instruct -f Modelfile-databot-instruct",
                    "Creating databot-instruct model"
                )
                if not success:
                    print("❌ Failed to create databot-instruct model")
                    return False
            else:
                print("⚠️ Modelfile-databot-instruct not found, will create it...")
                return False
        else:
            print("✅ databot-instruct model already available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up models: {e}")
        return False

def test_system():
    """Run basic system tests"""
    print("🧪 Running basic system tests...")
    
    # Test Ollama connection
    if check_ollama_running():
        print("✅ Ollama server connection test passed")
    else:
        print("❌ Ollama server connection test failed")
        return False
    
    # Test model availability
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags")
        models = response.json()
        model_names = [model['name'] for model in models.get('models', [])]
        
        if 'databot-instruct:latest' in model_names:
            print("✅ databot-instruct model test passed")
        else:
            print("❌ databot-instruct model not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    try:
        import pandas as pd
        
        # Create sample sales data
        data = {
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'sales': [100 + i*2 + (i%7)*10 for i in range(100)],
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['A', 'B', 'C'] * 33 + ['A'],
            'revenue': [1000 + i*20 + (i%7)*100 for i in range(100)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv('data/sample_sales_data.csv', index=False)
        print("✅ Created sample_sales_data.csv")
        
        # Create sample customer data
        customer_data = {
            'customer_id': range(1, 51),
            'age': [25 + i%40 for i in range(50)],
            'gender': ['M', 'F'] * 25,
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 10,
            'total_purchases': [100 + i*50 for i in range(50)]
        }
        
        df_customers = pd.DataFrame(customer_data)
        df_customers.to_csv('data/sample_customer_data.csv', index=False)
        print("✅ Created sample_customer_data.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

if __name__ == "__main__":
    main()