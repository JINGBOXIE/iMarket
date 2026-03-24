#!/bin/bash
git add .
git commit -m "Auto sync: $(date)"
git push origin main
echo "🚀 iMarket Pro 同步成功！"
