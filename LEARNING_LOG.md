📅 2026.2.26 | Git 推送流程与网络配置
🚀 从 0 推送代码的标准流程
Bash
git init                # 1. 初始化仓库（仅首次建库需要）
git add .               # 2. 添加所有文件至暂存区（注意 add 和 . 之间必须有空格）
git commit -m "备注信息"  # 3. 提交并添加代码变更备注
git branch -M main      # 4. 将主分支命名为 main
git push -u origin main # 5. 首次推送到远程仓库并绑定追踪关系
🌐 解决推送远程超时（配置终端代理）
终端默认不走系统 VPN，遇到端口超时需要单独给 Git 项目配置本地代理。

1. 给当前项目配置代理：

Bash
git config --local http.proxy http://127.0.0.1:端口号
git config --local https.proxy http://127.0.0.1:端口号
2. 取消当前项目的代理设置：

Bash
git config --local --unset http.proxy   # 注意 unset 前面是两个横杠 --
git config --local --unset https.proxy
