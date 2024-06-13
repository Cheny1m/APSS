import requests

# 设置 GitHub API 的基本 URL 和仓库信息
api_base_url = "https://api.github.com"
repo_owner = "your_username"
repo_name = "your_repository"

# 设置 GitHub 个人访问令牌(如果需要访问私有仓库)
access_token = "your_access_token"

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
    
    # 处理贡献者数据
    for contributor in contributors_data:
        username = contributor["login"]
        contributions = contributor["contributions"]
        print(f"用户名: {username}, 贡献次数: {contributions}")
else:
    print(f"获取贡献者数据失败,状态码: {response.status_code}")
