# Model Selection Documentation

## Selected Model
**Databot**

## Rationale
After thorough research and analysis, we have selected the Databot model for our data AI chatbot application for the following reasons:

1. **Size and Efficiency**: At 1.5B parameters, this model meets our requirement for a small, efficient model that can run locally with minimal hardware requirements. The model is approximately 1.1GB in size when quantized, making it suitable for embedding within the application.

2. **Performance**: According to Ollama's documentation, this model achieves performance comparable to OpenAI's o1-preview despite its small size. It has been specifically distilled from the larger DeepSeek-R1 model to maintain reasoning capabilities while reducing parameter count.

3. **Reasoning Capabilities**: The model excels at math, code, and reasoning tasks, which aligns perfectly with our requirements for complex data analysis and insight generation.

4. **Ollama Compatibility**: The model is officially supported by Ollama with 45.3M pulls, indicating strong community adoption and reliable integration.

5. **Licensing**: The model is available under the MIT License with Apache 2.0 components (from Qwen-2.5), allowing for commercial use and modifications, which is essential for our application's potential deployment scenarios.

6. **Distillation Approach**: The model was created by fine-tuning Qwen-2.5 with 800k samples curated with DeepSeek-R1, meaning it inherits the reasoning patterns of the larger model, which is advantageous for our data analysis requirements.

## Alternative Considered
**Qwen3 1.7B**

While Qwen3 1.7B was also considered as it meets our size requirements and offers tools capabilities, we selected Databot due to its specific optimization for reasoning tasks and slightly smaller size.

## Technical Specifications
- **Size**: 1.1GB (quantized)
- **Context Length**: 128K tokens
- **Input Type**: Text
- **Ollama Command**: `ollama run databot`

## Integration Plan
1. Install Ollama on the target system
2. Download and set up the phi4-mini-reasoning:latest model
3. Create a custom model with a specialized system prompt for data analysis
4. Implement RAG and vector embedding capabilities
5. Add API fallback options for more complex queries

This model selection provides an optimal balance between performance, size, and capabilities for our data AI chatbot application.
