import requests

# 设置 GitHub API 的基本 URL 和仓库信息
api_base_url = "https://api.github.com"
repo_owner = "Cheny1m"
repo_name = "APSS"

# 设置 GitHub 个人访问令牌（可选,如果你的仓库是私有的,则需要提供令牌）
access_token = "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"

# 构建 API 请求的 URL
contributors_url = f"{api_base_url}/repos/{repo_owner}/{repo_name}/contributors"

# 发送 GET 请求获取贡献者数据
headers = {
    "Authorization": f"token {access_token}"
} if access_token else {}
response = requests.get(contributors_url, headers=headers)

# 检查请求是否成功
if response.status_code == 200:
    contributors_data = response.json()
    
    # 生成 Markdown 格式的贡献者列表
    markdown_content = "## Contributors\n\n"
    for contributor in contributors_data:
        username = contributor["login"]
        profile_url = contributor["html_url"]
        avatar_url = contributor["avatar_url"]
        
        markdown_content += f"- [![{username}]({avatar_url}&s=50)]({profile_url})\n"
    
    # 将生成的 Markdown 内容写入 README.md 文件
    with open("README.md", "w") as file:
        file.write(markdown_content)
    
    print("Contributors list generated successfully!")
else:
    print(f"Failed to fetch contributors data. Status code: {response.status_code}")
