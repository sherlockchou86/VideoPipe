使用方法
命令行模式
BASH
# 分析单张图片
doubao_analyzer --api-key YOUR_API_KEY --image test.jpg

# 分析单个视频
doubao_analyzer --api-key YOUR_API_KEY --video test.mp4 --video-frames 8

# 批量分析文件夹
doubao_analyzer --api-key YOUR_API_KEY --folder ./media --file-type all --max-files 10

# 仅分析视频文件
doubao_analyzer --api-key YOUR_API_KEY --folder ./videos --file-type video

# 保存结果到文件
doubao_analyzer --api-key YOUR_API_KEY --folder ./media --output results.json