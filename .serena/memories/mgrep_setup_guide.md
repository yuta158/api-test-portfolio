# mgrep（Semantic Code Search）セットアップガイド

*最終更新: 2026年02月14日*

## 概要

mgrepは意味的関連性でコードを検索するセマンティックコード検索ツール。Mixedbread製。

**用途:**
- ローカル開発: 高度なコード検索（任意）
- CI/CD: 本リポジトリでは未統合（ローカル利用のみ）

**vs ast-grep:**
- **mgrep**: セマンティック検索、意味的関連性、文書構造探索推奨
- **ast-grep**: AST構造解析、構文的正確性、コード変更時必須

---

## インストール

### 方法1: package.json経由（推奨）

```bash
npm install  # devDependenciesに記載済み（@mixedbread/mgrep@0.1.13）
```

### 方法2: グローバルインストール

```bash
npm install -g @mixedbread/mgrep
```

---

## 認証設定

### ローカル開発: ブラウザ認証（推奨）

```bash
mgrep login
```

- ブラウザが自動起動
- Mixedbreadアカウントで認証（GitHub/Google OAuth対応）
- 認証情報は `~/.config/mgrep/` に保存

### CI/CD環境

このリポジトリでは mgrep を GitHub Actions に統合していません。CI で利用する場合は、対象ワークフローと Secrets の設定を別途検証してください。

---

## 使用例

### 基本検索

```bash
# セマンティック検索（意味的に関連するコードを検索）
mgrep search "API error handling" . --limit 5

# 検索結果の内容も表示
mgrep search "authentication" . --content

# 特定ディレクトリ内検索
mgrep search "database query" ./utils --limit 10
```

### 高度な使用

```bash
# 詳細ヘルプ
mgrep --help

# バージョン確認
mgrep --version
```

---

## トラブルシューティング

### `authentication failed`（ローカル）

**原因:** ブラウザ認証未完了

**対処:**
```bash
mgrep logout
mgrep login  # 再認証
```

### `command not found: mgrep`

**原因:** グローバルインストール未実行

**対処:**
```bash
npm install -g @mixedbread/mgrep
# または
npx @mixedbread/mgrep search "..." .
```

---

## 参考リンク

- [mgrep公式ドキュメント](https://github.com/mixedbread-ai/mgrep)
- [Mixedbread Dashboard](https://www.mixedbread.ai/)
- [ast-grep/ast-grep-mcp](https://github.com/ast-grep/ast-grep-mcp)（併用推奨）

---

## プロジェクト統合状況

**package.json:**
```json
"devDependencies": {
  "@mixedbread/mgrep": "^0.1.13"
}
```

**CI/CD統合:**
- 本リポジトリでは未統合（GitHub Actionsワークフローなし）

**.env.example:**
- `MXBAI_API_KEY` のプレースホルダーのみ（CI連携は未設定）
