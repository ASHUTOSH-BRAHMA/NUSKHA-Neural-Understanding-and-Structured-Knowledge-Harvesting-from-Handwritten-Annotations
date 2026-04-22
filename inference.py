model.eval()

with torch.no_grad():
    image = enhancer("sample.png").unsqueeze(0).to(device)
    output_ids = model.model.generate(image, max_length=512)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(result)
