"""utils.jsonplaceholder_client_sync モジュールの同期APIクライアントテスト


Note:
 - Create/Update/Delete は永続化されない。POST は常に ``id=101`` を返す。
"""

import json
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

from tests.constants import BASE_URL
from tests.unit.helpers import make_mock_user, mock_get_route
from utils.exceptions import APIHTTPError, APIJSONDecodeError, APIRetryError
from utils.jsonplaceholder_client_sync import SyncJSONPlaceholderClient

pytestmark = pytest.mark.unit


@respx.mock
def test_sync_health_check_success() -> None:
    route = respx.get(f"{BASE_URL}/users", params={"_limit": 1}).respond(
        json=[{"id": 1, "name": "User 1"}]
    )

    with SyncJSONPlaceholderClient() as client:
        result = client.health_check()

    assert result is True
    assert route.call_count == 1


@respx.mock
def test_sync_health_check_connection_error() -> None:
    route = respx.get(f"{BASE_URL}/users", params={"_limit": 1}).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with SyncJSONPlaceholderClient(retry_count=0) as client:
        result = client.health_check()

    assert result is False
    assert route.call_count == 1  # retry_count=0なのでリトライなし（1回のみ実行）


@respx.mock
def test_sync_health_check_log_structure() -> None:
    respx.get(f"{BASE_URL}/users", params={"_limit": 1}).mock(
        side_effect=httpx.ConnectError("Connection refused to secret-host.internal")
    )

    with SyncJSONPlaceholderClient(retry_count=0) as client:
        with patch.object(client, "logger") as mock_logger:
            result = client.health_check()

    assert result is False
    # warning は request_error と health_check_failed の2回呼ばれる
    assert mock_logger.warning.call_count == 2
    # health_check_failed の呼び出しを抽出
    health_check_call = next(
        (c for c in mock_logger.warning.call_args_list if c[0][0] == "health_check_failed"),
        None,
    )
    assert health_check_call is not None, "health_check_failed ログが出力されていない"
    # 必須フィールドの検証
    assert health_check_call[1]["error_type"] == "APIRetryError"
    assert health_check_call[1]["endpoint"] == "/users"
    # セキュリティ: error フィールド省略（_classify_error と同方針）
    assert "error" not in health_check_call[1]
    # all_retries_failed の error フィールド省略検証（機密情報保護）
    all_retries_call = next(
        (c for c in mock_logger.error.call_args_list if c[0][0] == "all_retries_failed"),
        None,
    )
    assert all_retries_call is not None, "all_retries_failed ログが出力されていること"
    assert "error" not in all_retries_call[1]


@pytest.mark.parametrize(
    "limit,expected_count",
    [(2, 2), (None, 5), (0, 0), (100, 5)],
    ids=["with_limit", "no_limit", "zero_limit", "excessive_limit"],
)
@respx.mock
def test_sync_get_posts(limit: int | None, expected_count: int) -> None:
    all_posts = [
        {"id": i, "userId": 1, "title": f"Post {i}", "body": f"Content {i}"} for i in range(1, 6)
    ]

    # limitパラメータに応じてモックデータを設定
    if limit is None:
        mock_data = all_posts
    elif limit == 0:
        # API仕様: _limit=0は空配列[]を返却
        mock_data = []
    else:
        mock_data = all_posts[:limit]

    # クエリパラメータ検証: limitが指定された場合はparams=でマッチ
    params = {"_limit": limit} if limit is not None else None
    route = mock_get_route(f"{BASE_URL}/posts", params, mock_data)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_posts(limit=limit)

    assert len(result) == expected_count
    assert [post.model_dump(by_alias=True) for post in result] == mock_data
    assert route.call_count == 1  # GETリクエストが1回のみ発行されたことを確認


@pytest.mark.parametrize(
    "user_id,expected_count",
    [
        (None, 5),  # user_id=Noneで全投稿取得
        (1, 2),  # user_id=1でユーザー1の投稿のみ
        (2, 1),  # user_id=2でユーザー2の投稿のみ
        (999, 0),  # user_id=999で存在しないユーザー
    ],
    ids=["no_filter", "user_1", "user_2", "nonexistent_user"],
)
@respx.mock
def test_sync_get_posts_user_filter(user_id: int | None, expected_count: int) -> None:
    all_posts = [
        {"id": 1, "userId": 1, "title": "Post 1", "body": "Content 1"},
        {"id": 2, "userId": 2, "title": "Post 2", "body": "Content 2"},
        {"id": 3, "userId": 1, "title": "Post 3", "body": "Content 3"},
        {"id": 4, "userId": 3, "title": "Post 4", "body": "Content 4"},
        {"id": 5, "userId": 3, "title": "Post 5", "body": "Content 5"},
    ]

    # user_idパラメータに応じてモックデータを設定
    if user_id is None:
        mock_data = all_posts
    elif user_id == 999:
        mock_data = []
    else:
        mock_data = [p for p in all_posts if p["userId"] == user_id]

    # クエリパラメータ検証: user_idが指定された場合はparams=でマッチ
    params = {"userId": user_id} if user_id is not None else None
    route = mock_get_route(f"{BASE_URL}/posts", params, mock_data)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_posts(user_id=user_id)

    assert len(result) == expected_count
    assert [post.model_dump(by_alias=True) for post in result] == mock_data
    assert route.call_count == 1  # GETリクエストが1回のみ発行されたことを確認
    if user_id is not None and user_id != 999:
        assert all(post.user_id == user_id for post in result)


@pytest.mark.parametrize(
    "limit,user_id,expected_error",
    [
        (-1, None, "limit must be >= 0"),
        (-100, None, "limit must be >= 0"),
        (None, 0, "user_id must be >= 1"),
        (None, -1, "user_id must be >= 1"),
        (-1, 0, "limit must be >= 0"),
    ],
    ids=[
        "negative_limit",
        "very_negative_limit",
        "zero_user_id",
        "negative_user_id",
        "both_invalid_limit_first",
    ],
)
def test_sync_get_posts_validation_error(
    limit: int | None, user_id: int | None, expected_error: str
) -> None:
    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            client.get_posts(limit=limit, user_id=user_id)


@respx.mock
def test_sync_get_post_success() -> None:
    post_id = 1
    expected_post = {"id": 1, "userId": 1, "title": "Test Post", "body": "Test Content"}

    route = respx.get(f"{BASE_URL}/posts/{post_id}").respond(json=expected_post)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_post(post_id)

        assert result.model_dump(by_alias=True) == expected_post
        assert result.id == post_id
    assert route.call_count == 1  # GETリクエストが1回のみ発行されたことを確認


@respx.mock
def test_sync_create_post() -> None:
    title = "New Post"
    body = "This is a new post content"
    user_id = 1

    expected_response = {
        "id": 101,  # サーバーが生成したID
        "userId": user_id,
        "title": title,
        "body": body,
    }

    route = respx.post(f"{BASE_URL}/posts").respond(status_code=201, json=expected_response)

    with SyncJSONPlaceholderClient() as client:
        result = client.create_post(title=title, body=body, user_id=user_id)

    # リクエストボディ検証: create_post()が正しいフィールドを送信しているか確認
    assert route.call_count == 1
    request_body = json.loads(route.calls[0].request.content)
    assert request_body["title"] == title
    assert request_body["body"] == body
    assert request_body["userId"] == user_id

    # レスポンス検証
    assert result.id == 101
    assert result.user_id == user_id
    assert result.title == title
    assert result.body == body


@respx.mock
@patch("utils.jsonplaceholder_base_sync.exponential_backoff_with_jitter", return_value=0.0)
def test_sync_create_post_retries_when_explicitly_opted_in(mock_backoff: Mock) -> None:
    """ドメインのPOSTでも明示的なオプトインがbase clientへ伝播する。"""
    route = respx.post(f"{BASE_URL}/posts")
    route.side_effect = [
        httpx.Response(502),
        httpx.Response(
            201,
            json={"id": 101, "userId": 1, "title": "created", "body": "content"},
        ),
    ]

    with SyncJSONPlaceholderClient(retry_count=2) as client:
        result = client.create_post(
            title="created",
            body="content",
            user_id=1,
            retry_non_idempotent=True,
        )

    assert route.call_count == 2
    assert result.id == 101
    assert mock_backoff.call_count == 1


@pytest.mark.parametrize(
    "user_id,completed,limit,expected_count",
    [
        (1, True, 5, 2),
        (1, None, None, 3),
        (None, False, 10, 2),
        (None, None, None, 5),
    ],
    ids=["all_params", "user_id_only", "completed_and_limit", "no_params"],
)
@respx.mock
def test_sync_get_todos(
    user_id: int | None,
    completed: bool | None,
    limit: int | None,
    expected_count: int,
) -> None:
    all_todos = [
        {"id": 1, "userId": 1, "title": "Todo 1", "completed": True},
        {"id": 2, "userId": 1, "title": "Todo 2", "completed": False},
        {"id": 3, "userId": 1, "title": "Todo 3", "completed": True},
        {"id": 4, "userId": 2, "title": "Todo 4", "completed": True},
        {"id": 5, "userId": 2, "title": "Todo 5", "completed": False},
    ]

    # パラメータに応じてフィルタされたモックデータを作成
    filtered_todos = all_todos
    if user_id is not None:
        filtered_todos = [t for t in filtered_todos if t["userId"] == user_id]
    if completed is not None:
        filtered_todos = [t for t in filtered_todos if t["completed"] == completed]
    if limit is not None:
        filtered_todos = filtered_todos[:limit]

    # クエリパラメータ検証: 指定されたパラメータのみparams=でマッチ
    # dict comprehensionでNoneを除外（completed=Falseは有効なパラメータとして保持）
    params = {
        k: v
        for k, v in {"userId": user_id, "completed": completed, "_limit": limit}.items()
        if v is not None
    } or None
    route = mock_get_route(f"{BASE_URL}/todos", params, filtered_todos)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_todos(user_id=user_id, completed=completed, limit=limit)

    assert len(result) == expected_count
    assert [todo.model_dump(by_alias=True) for todo in result] == filtered_todos
    assert route.call_count == 1  # GETリクエストが1回のみ発行されたことを確認

    # completed パラメータのエンコード検証
    # httpxはFalseを"false"、Trueを"true"にエンコードする（小文字）
    if completed is not None:
        assert route.calls[0].request.url.params.get("completed") == str(completed).lower(), (
            f"completed={completed} はhttpxにより '{str(completed).lower()}' にエンコードされるべき"
        )


@pytest.mark.parametrize(
    "limit,user_id,expected_error",
    [
        (-1, None, "limit must be >= 0"),
        (-100, None, "limit must be >= 0"),
        (None, 0, "user_id must be >= 1"),
        (None, -1, "user_id must be >= 1"),
        (-1, 0, "limit must be >= 0"),
    ],
    ids=[
        "negative_limit",
        "very_negative_limit",
        "zero_user_id",
        "negative_user_id",
        "both_invalid_limit_first",
    ],
)
def test_sync_get_todos_validation_error(
    limit: int | None, user_id: int | None, expected_error: str
) -> None:
    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            client.get_todos(limit=limit, user_id=user_id)


@pytest.mark.parametrize(
    "user_id,expected_count",
    [(1, 2), (None, 5), (2, 1)],
    ids=["user_id_1", "no_user_id", "user_id_2"],
)
@respx.mock
def test_sync_get_albums(user_id: int | None, expected_count: int) -> None:
    # モックデータ（5件のアルバム、複数ユーザー）
    all_albums = [
        {"id": 1, "userId": 1, "title": "Album 1"},
        {"id": 2, "userId": 1, "title": "Album 2"},
        {"id": 3, "userId": 2, "title": "Album 3"},
        {"id": 4, "userId": 3, "title": "Album 4"},
        {"id": 5, "userId": 3, "title": "Album 5"},
    ]

    # パラメータに応じてフィルタ
    if user_id is not None:
        mock_data = [a for a in all_albums if a["userId"] == user_id]
    else:
        mock_data = all_albums

    # クエリパラメータ検証: user_idが指定された場合はparams=でマッチ
    params = {"userId": user_id} if user_id is not None else None
    route = mock_get_route(f"{BASE_URL}/albums", params, mock_data)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_albums(user_id=user_id)

    assert len(result) == expected_count
    assert [album.model_dump(by_alias=True) for album in result] == mock_data
    assert route.call_count == 1  # GETリクエストが1回のみ発行されたことを確認


@pytest.mark.parametrize(
    "user_id,expected_error",
    [
        (0, "user_id must be >= 1"),
        (-1, "user_id must be >= 1"),
    ],
    ids=["zero_user_id", "negative_user_id"],
)
def test_sync_get_albums_validation_error(user_id: int, expected_error: str) -> None:
    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            client.get_albums(user_id=user_id)


@pytest.mark.parametrize(
    "album_id,expected_count",
    [(1, 2), (None, 6), (2, 1)],
    ids=["album_id_1", "no_album_id", "album_id_2"],
)
@respx.mock
def test_sync_get_photos(album_id: int | None, expected_count: int) -> None:
    # モックデータ（6件の写真、複数アルバム）
    all_photos = [
        {
            "id": 1,
            "albumId": 1,
            "title": "Photo 1",
            "url": "https://example.com/1.jpg",
            "thumbnailUrl": "https://example.com/1-thumb.jpg",
        },
        {
            "id": 2,
            "albumId": 1,
            "title": "Photo 2",
            "url": "https://example.com/2.jpg",
            "thumbnailUrl": "https://example.com/2-thumb.jpg",
        },
        {
            "id": 3,
            "albumId": 2,
            "title": "Photo 3",
            "url": "https://example.com/3.jpg",
            "thumbnailUrl": "https://example.com/3-thumb.jpg",
        },
        {
            "id": 4,
            "albumId": 3,
            "title": "Photo 4",
            "url": "https://example.com/4.jpg",
            "thumbnailUrl": "https://example.com/4-thumb.jpg",
        },
        {
            "id": 5,
            "albumId": 3,
            "title": "Photo 5",
            "url": "https://example.com/5.jpg",
            "thumbnailUrl": "https://example.com/5-thumb.jpg",
        },
        {
            "id": 6,
            "albumId": 3,
            "title": "Photo 6",
            "url": "https://example.com/6.jpg",
            "thumbnailUrl": "https://example.com/6-thumb.jpg",
        },
    ]

    # パラメータに応じてフィルタとエンドポイントを設定
    if album_id is not None:
        mock_data = [p for p in all_photos if p["albumId"] == album_id]
        route = respx.get(f"{BASE_URL}/albums/{album_id}/photos").respond(json=mock_data)
    else:
        mock_data = all_photos
        route = respx.get(f"{BASE_URL}/photos").respond(json=mock_data)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_photos(album_id=album_id)

    assert len(result) == expected_count
    assert [photo.model_dump(by_alias=True) for photo in result] == mock_data
    assert route.call_count == 1


@respx.mock
def test_sync_get_comments_with_post_id() -> None:
    mock_comments = [
        {"id": 1, "postId": 1, "name": "Test Comment", "email": "test@example.com", "body": "Body"},
    ]
    route = respx.get(f"{BASE_URL}/posts/1/comments").respond(json=mock_comments)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_comments(post_id=1)

    assert route.call_count == 1
    assert [comment.model_dump(by_alias=True) for comment in result] == mock_comments


@respx.mock
def test_sync_get_comments_without_post_id() -> None:
    mock_comments = [
        {"id": 1, "postId": 1, "name": "Comment 1", "email": "a@b.com", "body": "Body 1"},
        {"id": 2, "postId": 2, "name": "Comment 2", "email": "c@d.com", "body": "Body 2"},
    ]
    route = mock_get_route(f"{BASE_URL}/comments", None, mock_comments)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_comments()

    assert route.call_count == 1
    assert [comment.model_dump(by_alias=True) for comment in result] == mock_comments


@pytest.mark.parametrize(
    "post_id",
    [0, -1, -100],
    ids=["post_id_zero", "post_id_negative", "post_id_large_negative"],
)
def test_sync_get_comments_invalid_post_id(post_id: int) -> None:
    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match="post_id must be >= 1"):
            client.get_comments(post_id=post_id)


@pytest.mark.parametrize(
    "album_id",
    [0, -1, -100],
    ids=["album_id_zero", "album_id_negative", "album_id_large_negative"],
)
def test_sync_get_photos_invalid_album_id(album_id: int) -> None:
    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match="album_id must be >= 1"):
            client.get_photos(album_id=album_id)


@respx.mock
def test_sync_patch_method() -> None:
    endpoint = "/todos/1"
    patch_data = {"completed": True}
    full_response = {"id": 1, "title": "Test Todo", "completed": True, "userId": 1}

    # PATCHレスポンスをモック化
    route = respx.patch(f"{BASE_URL}{endpoint}").respond(status_code=200, json=full_response)

    with SyncJSONPlaceholderClient() as client:
        result = client.update_todo(1, completed=True)

    # レスポンス検証
    assert result == full_response
    assert result["completed"] is True

    # リクエスト検証
    assert route.call_count == 1
    assert route.calls[0].request.method == "PATCH"

    # リクエストボディ検証
    request_body = json.loads(route.calls[0].request.content)
    assert request_body == patch_data


@respx.mock
def test_sync_update_post() -> None:
    updated_data = {"id": 1, "title": "Updated Title", "body": "Updated Body"}
    route = respx.put(f"{BASE_URL}/posts/1").respond(status_code=200, json=updated_data)

    with SyncJSONPlaceholderClient() as client:
        result = client.update_post(1, "Updated Title", "Updated Body")

    assert result == updated_data
    assert route.call_count == 1
    assert route.calls[0].request.method == "PUT"
    request_body = json.loads(route.calls[0].request.content)
    assert request_body == {"title": "Updated Title", "body": "Updated Body"}


@respx.mock
def test_sync_delete_post() -> None:
    route = respx.delete(f"{BASE_URL}/posts/1").respond(status_code=200)

    with SyncJSONPlaceholderClient() as client:
        result = client.delete_post(1)

    assert result is None
    assert route.call_count == 1


@respx.mock
def test_sync_create_user() -> None:
    user_data = {
        "name": "New Sync User",
        "email": "sync@example.com",
        "phone": "123-456-7890",
    }
    created_user = {"id": 101, **user_data}
    route = respx.post(f"{BASE_URL}/users").respond(status_code=201, json=created_user)

    with SyncJSONPlaceholderClient() as client:
        result = client.create_user(user_data)

    assert result == created_user
    assert route.call_count == 1
    assert route.calls[0].request.method == "POST"
    request_body = json.loads(route.calls[0].request.content)
    assert request_body == user_data


@respx.mock
def test_sync_get_user() -> None:
    """async 側 test_async_get_user と対を成す。"""
    mock_user = make_mock_user(
        1,
        name="Leanne Graham",
        username="Bret",
        email="sincere@april.biz",
        website="https://hildegard.org",
    )

    route = respx.get(f"{BASE_URL}/users/1").respond(json=mock_user)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_user(1)

    # async 版と同じく model_dump の往復で入れ子モデルまで含む全属性の契約を検証する
    assert result.model_dump(by_alias=True) == mock_user
    assert result.id == 1
    assert result.name == "Leanne Graham"
    assert result.email == "sincere@april.biz"
    assert route.call_count == 1


@respx.mock
def test_sync_get_users() -> None:
    mock_users = [
        make_mock_user(
            1,
            name="Leanne Graham",
            username="Bret",
            email="sincere@april.biz",
            website="https://hildegard.org",
        ),
        make_mock_user(
            2,
            name="Ervin Howell",
            username="Antonette",
            email="shanna@melissa.tv",
            website="https://anastasia.net",
        ),
    ]

    route = respx.get(f"{BASE_URL}/users").respond(json=mock_users)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_users()

    assert len(result) == 2
    # 全属性を検証し、フィールドマッピングの回帰（属性の取りこぼし）を防ぐ
    assert result[0].id == 1
    assert result[0].name == "Leanne Graham"
    assert result[0].username == "Bret"
    assert result[0].email == "sincere@april.biz"
    assert result[1].id == 2
    assert result[1].name == "Ervin Howell"
    assert result[1].username == "Antonette"
    assert route.call_count == 1


@respx.mock
def test_sync_get_todo() -> None:
    mock_todo = {"id": 1, "userId": 1, "title": "delectus aut autem", "completed": False}

    route = respx.get(f"{BASE_URL}/todos/1").respond(json=mock_todo)

    with SyncJSONPlaceholderClient() as client:
        result = client.get_todo(1)

    assert result.id == 1
    assert result.title == "delectus aut autem"
    assert result.completed is False
    assert route.call_count == 1


@respx.mock
def test_sync_create_todo() -> None:
    new_todo_response = {
        "id": 201,
        "title": "Buy groceries",
        "userId": 1,
        "completed": False,
    }

    route = respx.post(f"{BASE_URL}/todos").respond(status_code=201, json=new_todo_response)

    with SyncJSONPlaceholderClient() as client:
        result = client.create_todo(title="Buy groceries", user_id=1, completed=False)

        # レスポンス検証: userId -> user_id の alias マッピングと
        # completed フィールドまで含め、全属性の契約を検証する
        assert result.id == 201
        assert result.title == "Buy groceries"
        assert result.user_id == 1
        assert result.completed is False
    assert route.call_count == 1

    # リクエストボディ検証: title/userId/completedが正しく送信されたか
    request_body = json.loads(route.calls[0].request.content)
    assert request_body["title"] == "Buy groceries"
    assert request_body["userId"] == 1
    assert request_body["completed"] is False


@pytest.mark.parametrize(
    ("exception_class", "exception_args"),
    [
        pytest.param(SystemExit, (1,), id="SystemExit"),
        pytest.param(MemoryError, ("Out of memory",), id="MemoryError"),
        pytest.param(RecursionError, ("maximum recursion depth exceeded",), id="RecursionError"),
    ],
)
def test_sync_health_check_system_exception_propagates(
    exception_class: type[BaseException],
    exception_args: tuple[object, ...],
) -> None:
    """システム例外がhealth_checkのexcept APIClientErrorで握りつぶされないことを検証

    SyncAPIClient.health_check() は except APIClientError の前に
    SYNC_FATAL_EXCEPTIONS (SystemExit / MemoryError / RecursionError) を明示的に re-raise する。
    （KeyboardInterruptはpytest自体がSIGINTハンドラとして処理するためunitテストでの検証は省略）
    この設計により:
    - SystemExit: graceful shutdown シグナルがプロセス外へ正しく伝播
    - MemoryError: K8s OOMKilled 検知が遅延しない
    - RecursionError: スタック枯渇が except APIClientError に隠蔽されず fail-fast 伝播する
    （非同期版 parametrize との対称性を回復）

    回帰テスト: except 節の順序変更や削除による退行を検出。
    """
    with SyncJSONPlaceholderClient() as client:
        with patch.object(client, "get", side_effect=exception_class(*exception_args)):
            with pytest.raises(exception_class):
                client.health_check()


@respx.mock
def test_sync_update_post_404_error() -> None:
    route = respx.put(f"{BASE_URL}/posts/99999").respond(
        status_code=404,
        json={"error": "Post not found"},
    )

    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(APIHTTPError) as exc_info:
            client.update_post(99999, "Title", "Body")
        assert exc_info.value.status_code == 404

    assert route.call_count == 1  # 4xxはリトライせず即失敗


@respx.mock
@patch("utils.jsonplaceholder_base_sync.exponential_backoff_with_jitter", return_value=0.0)
def test_sync_delete_post_500_error(mock_backoff: Mock) -> None:
    route = respx.delete(f"{BASE_URL}/posts/1").respond(
        status_code=500,
        json={"error": "Internal server error"},
    )

    with SyncJSONPlaceholderClient(retry_count=3) as client:
        with pytest.raises(APIRetryError):
            client.delete_post(1)

    assert route.call_count == 4  # retry_count=3 → 初回 + リトライ3回


@pytest.mark.parametrize(
    ("method_name", "http_verb", "path", "call_args"),
    [
        ("update_post", "put", "/posts/1", (1, "Title", "Body")),
        ("create_user", "post", "/users", ({"name": "New User"},)),
        ("update_todo", "patch", "/todos/1", (1,)),
    ],
)
@respx.mock
def test_sync_dict_returning_methods_reject_non_object_json(
    method_name: str,
    http_verb: str,
    path: str,
    call_args: tuple[object, ...],
) -> None:
    """2xx でも非オブジェクト JSON なら APIJSONDecodeError を送出する。

    ``dict[str, Any]`` の戻り値アノテーションは実行時に検証されないため、
    契約違反が APIClientError 階層を迂回して呼び出し側に漏れないことを保証する。
    """
    route = getattr(respx, http_verb)(f"{BASE_URL}{path}").respond(
        status_code=200,
        json=["unexpected", "array"],
    )

    with SyncJSONPlaceholderClient() as client:
        with pytest.raises(APIJSONDecodeError) as exc_info:
            getattr(client, method_name)(*call_args)

    assert "Expected object JSON response" in str(exc_info.value)
    assert "list" in str(exc_info.value)
    assert route.call_count == 1
