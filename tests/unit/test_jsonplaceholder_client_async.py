"""utils.jsonplaceholder_client_async モジュールの非同期APIクライアントテスト"""

import asyncio
import json
import time
from typing import Any, TypedDict
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx
from httpx import Response
from structlog.testing import capture_logs

from models.responses import Album, Photo, Post, Todo, User
from tests.constants import BASE_URL
from tests.unit.helpers import assert_warning_log_count, make_canonical_user
from utils.exceptions import (
    APIClientError,
    APIHTTPError,
    APIJSONDecodeError,
    APIRetryError,
)
from utils.jsonplaceholder_client_async import (
    MAX_LOGGED_FAILURE_DETAILS,
    AsyncJSONPlaceholderClient,
    _run_rolling_window,
)

pytestmark = pytest.mark.unit


class PostData(TypedDict):
    """投稿データの型定義（Dict構造を明確化）"""

    id: int
    title: str
    body: str
    userId: int


@pytest.fixture
def sample_post_data() -> PostData:
    """テスト用投稿データ"""
    return {
        "id": 101,
        "title": "Test Title",
        "body": "Test Body",
        "userId": 1,
    }


@pytest.fixture
def sample_user_data():
    """テスト用ユーザーデータ"""
    return {
        "id": 1,
        "name": "Leanne Graham",
        "username": "Bret",
        "email": "Sincere@april.biz",
        "address": {
            "street": "Kulas Light",
            "suite": "Apt. 556",
            "city": "Gwenborough",
            "zipcode": "92998-3874",
            "geo": {"lat": "-37.3159", "lng": "81.1496"},
        },
        "phone": "1-770-736-8031 x56442",
        "website": "https://hildegard.org",
        "company": {
            "name": "Romaguera-Crona",
            "catchPhrase": "Multi-layered client-server neural-net",
            "bs": "harness real-time e-markets",
        },
    }


@pytest.fixture
def sample_users_list():
    """テスト用ユーザーリスト"""
    return [
        {
            "id": 1,
            "name": "Leanne Graham",
            "username": "Bret",
            "email": "Sincere@april.biz",
            "address": {
                "street": "Kulas Light",
                "suite": "Apt. 556",
                "city": "Gwenborough",
                "zipcode": "92998-3874",
                "geo": {"lat": "-37.3159", "lng": "81.1496"},
            },
            "phone": "1-770-736-8031 x56442",
            "website": "https://hildegard.org",
            "company": {
                "name": "Romaguera-Crona",
                "catchPhrase": "Multi-layered client-server neural-net",
                "bs": "harness real-time e-markets",
            },
        },
        {
            "id": 2,
            "name": "Ervin Howell",
            "username": "Antonette",
            "email": "Shanna@melissa.tv",
            "address": {
                "street": "Victor Plains",
                "suite": "Suite 879",
                "city": "Wisokyburgh",
                "zipcode": "90566-7771",
                "geo": {"lat": "-43.9509", "lng": "-34.4618"},
            },
            "phone": "010-692-6593 x09125",
            "website": "https://anastasia.net",
            "company": {
                "name": "Deckow-Crist",
                "catchPhrase": "Proactive didactic contingency",
                "bs": "synergize scalable supply-chains",
            },
        },
        {
            "id": 3,
            "name": "Clementine Bauch",
            "username": "Samantha",
            "email": "Nathan@yesenia.net",
            "address": {
                "street": "Douglas Extension",
                "suite": "Suite 847",
                "city": "McKenziehaven",
                "zipcode": "59590-4157",
                "geo": {"lat": "-68.6102", "lng": "-47.0653"},
            },
            "phone": "1-463-123-4447",
            "website": "https://ramiro.info",
            "company": {
                "name": "Romaguera-Jacobson",
                "catchPhrase": "Face to face bifurcated interface",
                "bs": "e-enable strategic applications",
            },
        },
    ]


@respx.mock
async def test_async_create_post(sample_post_data: PostData) -> None:
    route = respx.post(f"{BASE_URL}/posts").respond(
        status_code=201,
        json=dict(sample_post_data),
    )

    async with AsyncJSONPlaceholderClient() as client:
        post = await client.create_post("Test Title", "Test Body", 1)

    assert route.call_count == 1
    request_body = json.loads(route.calls[0].request.content)
    assert request_body["title"] == "Test Title"
    assert request_body["body"] == "Test Body"
    assert request_body["userId"] == 1

    assert post.title == "Test Title"
    assert post.body == "Test Body"
    assert post.user_id == 1
    assert post.id == 101


@respx.mock
@patch("utils.jsonplaceholder_base_async.exponential_backoff_with_jitter", return_value=0.0)
async def test_async_create_post_retries_when_explicitly_opted_in(mock_backoff: Mock) -> None:
    """ドメインのPOSTでも明示的なオプトインがbase clientへ伝播する。"""
    route = respx.post(f"{BASE_URL}/posts")
    route.side_effect = [
        httpx.Response(502),
        httpx.Response(
            201,
            json={"id": 101, "userId": 1, "title": "created", "body": "content"},
        ),
    ]

    async with AsyncJSONPlaceholderClient(retry_count=2) as client:
        result = await client.create_post(
            title="created",
            body="content",
            user_id=1,
            retry_non_idempotent=True,
        )

    assert route.call_count == 2
    assert result.id == 101
    assert mock_backoff.call_count == 1


@respx.mock
async def test_async_update_post() -> None:
    updated_data = {
        "id": 1,
        "title": "Updated Title",
        "body": "Updated Body",
        "userId": 1,
    }
    route = respx.put(f"{BASE_URL}/posts/1").respond(
        status_code=200,
        json=updated_data,
    )

    async with AsyncJSONPlaceholderClient() as client:
        post = await client.update_post(1, "Updated Title", "Updated Body")

    # update_post は title と body のみ送信（create_post の userId とは異なり含まない）
    assert route.call_count == 1
    request_body = json.loads(route.calls[0].request.content)
    assert request_body["title"] == "Updated Title"
    assert request_body["body"] == "Updated Body"
    assert "userId" not in request_body  # PUT は部分更新: userId は送信しない

    assert post["id"] == 1
    assert post["title"] == "Updated Title"
    assert post["body"] == "Updated Body"


@respx.mock
async def test_async_delete_post() -> None:
    route = respx.delete(f"{BASE_URL}/posts/1").respond(status_code=200)

    async with AsyncJSONPlaceholderClient() as client:
        # delete_post() は None を返す: 型宣言はランタイム動作を保証しないため
        # 実際の戻り値を明示的に検証する（204 No Content → None 変換の保証）
        result = await client.delete_post(1)  # type: ignore[func-returns-value]

    assert route.call_count == 1
    assert result is None


@respx.mock
async def test_async_crud(sample_post_data: PostData) -> None:

    post_route = respx.post(f"{BASE_URL}/posts").respond(
        status_code=201,
        json=dict(sample_post_data),
    )

    get_route = respx.get(f"{BASE_URL}/posts").respond(
        status_code=200,
        json=[dict(sample_post_data)],
    )

    updated_data = {
        "id": 101,
        "title": "Updated",
        "body": "Updated Body",
        "userId": 1,
    }
    put_route = respx.put(f"{BASE_URL}/posts/101").respond(
        status_code=200,
        json=updated_data,
    )

    delete_route = respx.delete(f"{BASE_URL}/posts/101").respond(status_code=200)

    async with AsyncJSONPlaceholderClient() as client:
        post = await client.create_post("Test Title", "Test Body", 1)
        post_id = post.id
        assert post_id == 101
        assert post.title == "Test Title"

        retrieved_posts = await client.get_posts(limit=None)
        assert len(retrieved_posts) > 0
        assert retrieved_posts[0].id == post_id

        updated = await client.update_post(post_id, "Updated", "Updated Body")
        assert updated["id"] == post_id
        assert updated["title"] == "Updated"

        delete_result = await client.delete_post(post_id)  # type: ignore[func-returns-value]
        assert delete_result is None

    assert post_route.call_count == 1
    assert get_route.call_count == 1
    assert put_route.call_count == 1
    assert delete_route.call_count == 1


@respx.mock
async def test_async_create_post_400_error() -> None:
    route = respx.post(f"{BASE_URL}/posts").respond(
        status_code=400,
        json={"error": "Invalid request body"},
    )

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(APIHTTPError) as exc_info:
            await client.create_post("", "", 0)
        assert exc_info.value.status_code == 400

    assert route.call_count == 1


@respx.mock
async def test_async_update_post_404_error() -> None:
    route = respx.put(f"{BASE_URL}/posts/99999").respond(
        status_code=404,
        json={"error": "Post not found"},
    )

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(APIHTTPError) as exc_info:
            await client.update_post(99999, "Title", "Body")
        assert exc_info.value.status_code == 404

    assert route.call_count == 1


@respx.mock
@patch("utils.jsonplaceholder_base_async.exponential_backoff_with_jitter", return_value=0.0)
async def test_async_delete_post_500_error(mock_backoff: Mock) -> None:
    route = respx.delete(f"{BASE_URL}/posts/1").respond(
        status_code=500,
        json={"error": "Internal server error"},
    )

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(APIRetryError):
            await client.delete_post(1)

    assert route.call_count == 4


@respx.mock
async def test_async_get_user(sample_user_data):

    route = respx.get(f"{BASE_URL}/users/1").respond(json=sample_user_data)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_user(1)

        assert result.model_dump(by_alias=True) == sample_user_data
        assert result.id == 1
        assert result.name == "Leanne Graham"
        assert result.email == "Sincere@april.biz"

    assert route.call_count == 1


@respx.mock
async def test_async_concurrent_requests(sample_users_list):

    route_user1 = respx.get(f"{BASE_URL}/users/1").respond(json=sample_users_list[0])
    route_user2 = respx.get(f"{BASE_URL}/users/2").respond(json=sample_users_list[1])
    route_user3 = respx.get(f"{BASE_URL}/users/3").respond(json=sample_users_list[2])

    async with AsyncJSONPlaceholderClient() as client:
        user_ids = [1, 2, 3]
        tasks = [client.get_user(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(isinstance(result, User) for result in results)
        assert results[0].id == 1
        assert results[1].id == 2
        assert results[2].id == 3

    assert route_user1.call_count == 1
    assert route_user2.call_count == 1
    assert route_user3.call_count == 1


@respx.mock
async def test_async_multiple_users_with_rolling_window():
    """
    respx はリクエスト時刻を記録しないため、このテストでは戻り値だけを検証する。

    max_concurrent の実測は test_rolling_window_respects_max_concurrent で行う。
    """

    routes = {}
    for i in [1, 2, 3, 4, 5]:
        routes[i] = respx.get(f"{BASE_URL}/users/{i}").respond(json=make_canonical_user(i))

    async with AsyncJSONPlaceholderClient() as client:
        results = await client.get_multiple_users([1, 2, 3, 4, 5], max_concurrent=2)

        assert len(results) == 5
        assert all(isinstance(result, User) for result in results)
        assert results[0].id == 1
        assert results[4].id == 5

    assert all(r.call_count == 1 for r in routes.values())


@respx.mock
async def test_rolling_window_respects_max_concurrent():
    """
    asyncio.sleep(0) で event loop に制御を返し、rolling window による
    実行中タスク数の上限を観測する。
    """
    max_concurrent_observed = 0
    current_concurrent = 0

    original_get_user = AsyncJSONPlaceholderClient.get_user

    async def spy_get_user(self: AsyncJSONPlaceholderClient, user_id: int) -> User:
        nonlocal max_concurrent_observed, current_concurrent
        current_concurrent += 1
        max_concurrent_observed = max(max_concurrent_observed, current_concurrent)

        await asyncio.sleep(0)
        try:
            result = await original_get_user(self, user_id)
        finally:
            current_concurrent -= 1
        return result

    for i in [1, 2, 3, 4, 5]:
        respx.get(f"{BASE_URL}/users/{i}").respond(json={"id": i, "name": f"User {i}"})

    with patch.object(AsyncJSONPlaceholderClient, "get_user", spy_get_user):
        async with AsyncJSONPlaceholderClient() as client:
            await client.get_multiple_users([1, 2, 3, 4, 5], max_concurrent=2)

    assert max_concurrent_observed == 2, (
        f"同時実行数2を期待 (max_concurrent=2, 実際: {max_concurrent_observed})"
    )


@respx.mock
async def test_partial_failure_graceful_degradation():

    route1 = respx.get(f"{BASE_URL}/users/1").respond(json=make_canonical_user(1))
    route2 = respx.get(f"{BASE_URL}/users/2").respond(status_code=500)
    route3 = respx.get(f"{BASE_URL}/users/3").respond(json=make_canonical_user(3))
    route4 = respx.get(f"{BASE_URL}/users/4").respond(status_code=500)
    route5 = respx.get(f"{BASE_URL}/users/5").respond(json=make_canonical_user(5))

    # retry_count=0: リトライなし設定でgraceful degradationのみ検証（リトライ挙動は別テストで担保）
    with capture_logs() as log_output:
        async with AsyncJSONPlaceholderClient(retry_count=0) as client:
            results = await client.get_multiple_users([1, 2, 3, 4, 5], max_concurrent=2)

    assert len(results) == 3, f"Expected 3 successful results, got {len(results)}"
    assert all(isinstance(r, User) for r in results)

    result_ids = [r.id for r in results]
    assert sorted(result_ids) == [1, 3, 5], f"Expected IDs [1,3,5], got {result_ids}"

    # 全5エンドポイントが各1回ずつ呼ばれたことを確認（retry_count=0のため決定論的に==1）
    assert route1.call_count == 1
    assert route2.call_count == 1
    assert route3.call_count == 1
    assert route4.call_count == 1
    assert route5.call_count == 1

    assert_warning_log_count(log_output, "get_user_failed", 2)

    # セキュリティ: get_user_failed ログの構造検証（APIClientErrorメッセージはサニタイズ済み）
    for log in log_output:
        if log.get("event") == "get_user_failed":
            assert "error" not in log  # _classify_error と同方針で省略
            assert "error_type" in log


@respx.mock
async def test_all_requests_fail_returns_empty_list():

    routes = [respx.get(f"{BASE_URL}/users/{uid}").respond(status_code=500) for uid in [1, 2, 3]]

    # retry_count=0: リトライなし設定でgraceful degradationのみ検証（リトライ挙動は別テストで担保）
    with capture_logs() as log_output:
        async with AsyncJSONPlaceholderClient(retry_count=0) as client:
            results = await client.get_multiple_users([1, 2, 3], max_concurrent=2)

    assert results == [], f"Expected empty list, got {results}"
    assert isinstance(results, list), f"Expected list type, got {type(results)}"

    # 全3エンドポイントが各1回ずつ呼ばれたことを確認（retry_count=0のため決定論的に==1）
    assert all(r.call_count == 1 for r in routes)

    assert_warning_log_count(log_output, "get_user_failed", 3)

    # セキュリティ: get_user_failed ログの構造検証
    for log in log_output:
        if log.get("event") == "get_user_failed":
            assert "error" not in log  # _classify_error と同方針で省略
            assert "error_type" in log


@respx.mock
async def test_async_post_create_user():

    created_user = {
        "id": 101,
        "name": "New Async User",
        "email": "async@example.com",
        "phone": "123-456-7890",
    }
    route = respx.post(f"{BASE_URL}/users").respond(status_code=201, json=created_user)

    async with AsyncJSONPlaceholderClient() as client:
        user_data = {
            "name": "New Async User",
            "email": "async@example.com",
            "phone": "123-456-7890",
        }

        result = await client.create_user(user_data)

        assert result["id"] == 101
        assert result["name"] == "New Async User"
        assert result["email"] == "async@example.com"

    assert route.call_count == 1
    assert route.calls[0].request.method == "POST"

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["name"] == "New Async User"
    assert request_body["email"] == "async@example.com"
    assert request_body["phone"] == "123-456-7890"


@pytest.mark.parametrize("invalid_max_concurrent", [0, -1])
async def test_async_bulk_create_users_rejects_non_positive_max_concurrent(
    invalid_max_concurrent: int,
) -> None:
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
    ) as mock_create:
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
                await client.bulk_create_users(
                    [{"name": "A"}],
                    max_concurrent=invalid_max_concurrent,
                )

    mock_create.assert_not_called()


async def test_async_bulk_create_users_bounds_admission_window() -> None:
    first_finished = asyncio.Event()
    remaining_release = asyncio.Event()
    started_second = asyncio.Event()
    started_third = asyncio.Event()
    started_count = 0

    async def create_user(
        user_data: dict[str, object],
        *,
        retry_non_idempotent: bool = False,  # noqa: ARG001
    ) -> dict[str, object]:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            started_second.set()
        if started_count == 3:
            started_third.set()
        if user_data["id"] == 0:
            await first_finished.wait()
        else:
            await remaining_release.wait()
        return user_data

    users_to_create = [{"id": index} for index in range(5)]
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
        side_effect=create_user,
    ):
        async with AsyncJSONPlaceholderClient() as client:
            operation = asyncio.create_task(
                client.bulk_create_users(users_to_create, max_concurrent=2),
            )
            # timeout はハング検出の上限であり到達時間の期待値ではない。
            # 遅い CI ランナーでの誤検知を避けるため余裕を取る。
            await asyncio.wait_for(started_second.wait(), timeout=5.0)
            assert started_count == 2
            first_finished.set()
            await asyncio.wait_for(started_third.wait(), timeout=5.0)
            assert started_count == 3
            remaining_release.set()
            results = await operation

    assert [result["id"] for result in results] == [0, 1, 2, 3, 4]


async def test_async_multiple_users_cancels_and_awaits_pending_on_fatal() -> None:
    """収集対象外の例外で打ち切る際、保留タスクを cancel し終了まで待ってから伝播する契約。

    get_multiple_users は APIClientError のみを収集対象とするため、それ以外の例外は
    rolling window 内で即座に再送出される。後始末を怠るとタスクが未 await の
    まま破棄され、"Task was destroyed but it is pending" やイベントループ汚染として
    後続テストに漏れる。
    """
    blocked_started = asyncio.Event()
    blocked_cancelled = asyncio.Event()
    never_set = asyncio.Event()

    async def get_user(user_id: int) -> User:
        if user_id == 1:
            blocked_started.set()
            try:
                await never_set.wait()
            except asyncio.CancelledError:
                blocked_cancelled.set()
                raise
        await blocked_started.wait()
        raise RuntimeError("fatal failure outside the collected exception set")

    with patch.object(
        AsyncJSONPlaceholderClient,
        "get_user",
        new_callable=AsyncMock,
        side_effect=get_user,
    ):
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(RuntimeError, match="fatal failure"):
                await client.get_multiple_users([1, 2], max_concurrent=2)

    assert blocked_cancelled.is_set()


@pytest.mark.parametrize("invalid_max_concurrent", [0, -1])
async def test_async_multiple_users_rejects_non_positive_max_concurrent(
    invalid_max_concurrent: int,
) -> None:
    with patch.object(
        AsyncJSONPlaceholderClient,
        "get_user",
        new_callable=AsyncMock,
    ) as mock_get:
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
                await client.get_multiple_users(
                    [1],
                    max_concurrent=invalid_max_concurrent,
                )

    mock_get.assert_not_called()


@respx.mock
async def test_async_bulk_create_users():
    """
    同一URLへの並行POSTを区別するため、respx の side_effect を使う。
    """

    users_to_create = [
        {"name": "User 1", "email": "user1@test.com"},
        {"name": "User 2", "email": "user2@test.com"},
        {"name": "User 3", "email": "user3@test.com"},
    ]

    post_route = respx.post(f"{BASE_URL}/users")
    post_route.side_effect = [
        Response(201, json={"id": 101, "name": "User 1", "email": "user1@test.com"}),
        Response(201, json={"id": 102, "name": "User 2", "email": "user2@test.com"}),
        Response(201, json={"id": 103, "name": "User 3", "email": "user3@test.com"}),
    ]

    async with AsyncJSONPlaceholderClient() as client:
        results = await client.bulk_create_users(users_to_create)

    assert len(results) == 3
    assert all(result["id"] > 100 for result in results)

    assert post_route.call_count == 3

    # respxのside_effectリストは同一ルートへの呼び出し順に

    created_names = [result["name"] for result in results]
    assert "User 1" in created_names
    assert "User 2" in created_names
    assert "User 3" in created_names


@respx.mock
async def test_async_bulk_create_users_partial_failure_4xx_returns_only_successful():
    """
    return_exceptions=True で 4xx の APIHTTPError を吸収し、成功分だけ返す契約を固定する。

    4xx は即時失敗しリトライしないため、side_effect は到着順に 1:1 で消費され決定論的。
    """
    users_to_create = [
        {"name": "User 1", "email": "user1@test.com"},
        {"name": "User 2", "email": "user2@test.com"},
        {"name": "User 3", "email": "user3@test.com"},
    ]

    post_route = respx.post(f"{BASE_URL}/users")
    post_route.side_effect = [
        Response(201, json={"id": 101, "name": "User 1", "email": "user1@test.com"}),
        Response(422, json={"error": "Unprocessable Entity"}),
        Response(201, json={"id": 103, "name": "User 3", "email": "user3@test.com"}),
    ]

    async with AsyncJSONPlaceholderClient() as client:
        results = await client.bulk_create_users(users_to_create)

    # Note: respxのside_effectはリクエスト到着順（task作成順）に消費され決定論的
    assert len(results) == 2, "3件中1件(422)が失敗するため、成功件数は正確に2件であること"

    assert post_route.call_count == 3

    created_names = [result["name"] for result in results]
    assert "User 1" in created_names
    assert "User 3" in created_names
    assert "User 2" not in created_names


@respx.mock
async def test_async_bulk_create_users_partial_failure_4xx_log_structure():
    """
    PIIを含まない failed_details と、4xx固有の status_code を含むログ契約を固定する。
    """
    users_to_create = [
        {"name": "User 1", "email": "user1@test.com"},
        {"name": "User 2", "email": "user2@test.com"},
        {"name": "User 3", "email": "user3@test.com"},
    ]

    post_route = respx.post(f"{BASE_URL}/users")
    post_route.side_effect = [
        Response(201, json={"id": 101, "name": "User 1", "email": "user1@test.com"}),
        Response(422, json={"error": "Unprocessable Entity"}),
        Response(201, json={"id": 103, "name": "User 3", "email": "user3@test.com"}),
    ]

    with capture_logs() as log_output:
        async with AsyncJSONPlaceholderClient() as client:
            await client.bulk_create_users(users_to_create)

    partial_log = next(
        (log for log in log_output if log.get("event") == "bulk_create_partial_failure"),
        None,
    )
    assert partial_log is not None
    assert partial_log.get("log_level") == "warning"
    assert partial_log.get("failed_count") == 1
    assert partial_log.get("success_count") == 2

    failed_details = partial_log.get("failed_details", [])
    assert len(failed_details) == 1, "1件の失敗詳細が記録されていること"
    failed_item = failed_details[0]
    assert "error" not in failed_item  # _classify_error と同方針で省略
    assert "user_data" not in failed_item  # PII保護: emailは含まれないこと
    assert failed_item.get("error_type") == "APIHTTPError", "エラー種別が記録されていること"
    assert failed_item.get("status_code") == 422, "HTTPステータスコードが記録されていること"
    assert "index" in failed_item, "失敗ユーザーのインデックスが記録されていること"


@respx.mock
async def test_async_bulk_create_users_partial_failure_5xx_returns_only_successful() -> None:
    """
    5xx は APIRetryError へ変換される別経路でも、
    return_exceptions=True で成功分だけ返す契約を固定する。

    retry_count=0 で side_effect の到着順を決定論的にする。
    """
    users_to_create = [
        {"name": "User 1", "email": "user1@test.com"},
        {"name": "User 2", "email": "user2@test.com"},  # この1件を500エラーで失敗させる
        {"name": "User 3", "email": "user3@test.com"},
    ]

    # side_effectリスト要素数は並行タスク数と一致させること。
    # retry_count=0 の場合: 各タスクは初回(1) のみ = 3タスク × 1リクエスト = 3要素。
    # 不一致時はStopIterationが発生しデバッグが困難になる。
    post_route = respx.post(f"{BASE_URL}/users")
    post_route.side_effect = [
        Response(201, json={"id": 101, "name": "User 1", "email": "user1@test.com"}),
        Response(500, json={"error": "Internal Server Error"}),
        Response(201, json={"id": 103, "name": "User 3", "email": "user3@test.com"}),
    ]

    async with AsyncJSONPlaceholderClient(retry_count=0) as client:
        results = await client.bulk_create_users(users_to_create)

    assert len(results) == 2, (
        "3件中1件(5xx→APIRetryError)が失敗するため、成功件数は正確に2件であること"
    )

    assert post_route.call_count == 3

    created_names = [result["name"] for result in results]
    assert "User 1" in created_names
    assert "User 3" in created_names
    assert "User 2" not in created_names


@respx.mock
async def test_async_bulk_create_users_partial_failure_5xx_log_structure() -> None:
    """
    5xx の failed_details は status_code を持たず、PIIを含まないログ契約を固定する。
    """
    users_to_create = [
        {"name": "User 1", "email": "user1@test.com"},
        {"name": "User 2", "email": "user2@test.com"},  # この1件を500エラーで失敗させる
        {"name": "User 3", "email": "user3@test.com"},
    ]

    # side_effectリスト要素数は並行タスク数と一致させること。
    # retry_count=0 の場合: 各タスクは初回(1) のみ = 3タスク × 1リクエスト = 3要素。
    # 不一致時はStopIterationが発生しデバッグが困難になる。
    post_route = respx.post(f"{BASE_URL}/users")
    post_route.side_effect = [
        Response(201, json={"id": 101, "name": "User 1", "email": "user1@test.com"}),
        Response(500, json={"error": "Internal Server Error"}),
        Response(201, json={"id": 103, "name": "User 3", "email": "user3@test.com"}),
    ]

    with capture_logs() as log_output:
        async with AsyncJSONPlaceholderClient(retry_count=0) as client:
            await client.bulk_create_users(users_to_create)

    partial_log = next(
        (log for log in log_output if log.get("event") == "bulk_create_partial_failure"),
        None,
    )
    assert partial_log is not None
    assert partial_log.get("log_level") == "warning"
    assert partial_log.get("failed_count") == 1
    assert partial_log.get("success_count") == 2

    failed_details = partial_log.get("failed_details", [])
    assert len(failed_details) == 1
    failed_item = failed_details[0]
    assert "error" not in failed_item  # _classify_error と同方針で省略
    assert "user_data" not in failed_item  # PII保護: emailは含まれないこと
    assert failed_item.get("error_type") == "APIRetryError"
    assert "status_code" not in failed_item  # 5xxはAPIHTTPErrorでないためstatus_code不在
    assert "index" in failed_item, "失敗ユーザーのインデックスが記録されていること"


async def test_async_bulk_create_users_cancelled_error_propagates() -> None:
    """CancelledError は集約せず素の例外として伝播する契約を固定する。

    asyncio.TaskGroup が CancelledError を ExceptionGroup へ包まないのと同じ理由で、
    集約すると呼び出し側の ``except asyncio.CancelledError`` や asyncio.timeout が
    捕捉できず graceful shutdown が壊れる。
    """
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = asyncio.CancelledError()
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(asyncio.CancelledError):
                await client.bulk_create_users([{"name": "A"}, {"name": "B"}])

    assert mock_create.call_count == 2, "既定 max_concurrent では 2 件とも初期投入される"


async def test_async_bulk_create_users_single_cancelled_error_no_log() -> None:
    """
    単一キャンセルは通常の graceful shutdown としてログを出さず再送出する契約を固定する。
    """
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = asyncio.CancelledError()
        async with AsyncJSONPlaceholderClient() as client:
            with patch.object(client, "logger") as mock_logger:
                with pytest.raises(asyncio.CancelledError):
                    await client.bulk_create_users([{"name": "A"}])

                mock_logger.error.assert_not_called()


async def test_async_bulk_create_users_stops_admission_on_fatal() -> None:
    """fatal 例外の発生後、未投入ユーザーへの create_user 呼び出しを打ち切る契約。

    fatal を収集対象にすると rolling window が投入を続け、SIGTERM 後も残り全件へ
    POST が飛ぶ。呼び出し回数が window 幅で頭打ちになることで停止を検証する。
    """
    started_count = 0

    async def create_user(
        user_data: dict[str, object],
        *,
        retry_non_idempotent: bool = False,  # noqa: ARG001
    ) -> dict[str, object]:
        nonlocal started_count
        started_count += 1
        raise asyncio.CancelledError

    users_to_create = [{"id": index} for index in range(10)]
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
        side_effect=create_user,
    ):
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(asyncio.CancelledError):
                await client.bulk_create_users(users_to_create, max_concurrent=2)

    assert started_count == 2, "fatal 検知後に残り 8 件が投入されていないこと"


async def test_async_bulk_create_users_does_not_admit_after_same_batch_fatal() -> None:
    """同一完了バッチ内の fatal 検出前に後続ユーザーを投入しない契約。"""
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)

    async def create_user(
        user_data: dict[str, object],
        *,
        retry_non_idempotent: bool = False,  # noqa: ARG001
    ) -> dict[str, object]:
        if user_data["id"] == 1:
            raise asyncio.CancelledError
        return user_data

    users_to_create = [{"id": index} for index in range(4)]
    try:
        with patch.object(
            AsyncJSONPlaceholderClient,
            "create_user",
            new_callable=AsyncMock,
            side_effect=create_user,
        ) as mock_create:
            async with AsyncJSONPlaceholderClient() as client:
                with pytest.raises(asyncio.CancelledError):
                    await client.bulk_create_users(users_to_create, max_concurrent=2)

        assert mock_create.call_count == 2
    finally:
        loop.set_task_factory(previous_factory)


async def test_run_rolling_window_processes_batch_in_input_order() -> None:
    """同一バッチの完了タスクを入力インデックス順に処理する契約。

    ``asyncio.wait`` が返す ``done`` は set で反復順が不定。主例外の選択と追加 fatal の
    notes への記録を入力順で行うため、順序を固定しないと「どの例外が呼び出し側へ届くか」
    が実行ごとに変わり障害調査の再現性を欠く。
    """
    processed_order: list[int] = []

    async def operation(index: int) -> int:
        raise RuntimeError(str(index))

    def collect_exception(exc: BaseException) -> bool:
        processed_order.append(int(str(exc)))
        return True

    await _run_rolling_window(
        operation=operation,
        item_count=8,
        max_concurrent=8,
        collect_exception=collect_exception,
    )

    assert processed_order == list(range(8))


async def test_run_rolling_window_records_additional_fatal_exceptions() -> None:
    """同一バッチの通常 fatal は ExceptionGroup で診断情報を失わない。"""

    async def operation(index: int) -> int:
        if index == 0:
            raise MemoryError("first")
        raise RecursionError("second")

    with pytest.raises(ExceptionGroup) as exc_info:
        await _run_rolling_window(
            operation=operation,
            item_count=2,
            max_concurrent=2,
            collect_exception=lambda _: False,
        )

    assert [type(exception) for exception in exc_info.value.exceptions] == [
        MemoryError,
        RecursionError,
    ]


async def test_run_rolling_window_prioritizes_cancelled_error_over_fatal_exception() -> None:
    async def operation(index: int) -> int:
        if index == 0:
            raise MemoryError("first")
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await _run_rolling_window(
            operation=operation,
            item_count=2,
            max_concurrent=2,
            collect_exception=lambda _: False,
        )

    assert exc_info.value.__notes__ == ["Additional fatal exception at input index 0: MemoryError"]


@pytest.mark.parametrize(
    "fatal_exc",
    [MemoryError("OOM"), RecursionError("max depth")],
)
async def test_async_bulk_create_users_fatal_exception_propagates(
    fatal_exc: BaseException,
) -> None:
    """
    MemoryError / RecursionError を fatal exception として再送出し、
    OOM等を握りつぶさない契約を固定する。
    """
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.side_effect = fatal_exc
        async with AsyncJSONPlaceholderClient() as client:
            with pytest.raises(type(fatal_exc)):
                await client.bulk_create_users([{"name": "A"}])


@pytest.mark.parametrize("flag", [True, False])
async def test_async_bulk_create_users_forwards_retry_non_idempotent(flag: bool) -> None:
    """bulk 経由でも create_user の retry_non_idempotent がそのまま伝わる契約を固定する。"""
    with patch.object(
        AsyncJSONPlaceholderClient,
        "create_user",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = {"id": 1}
        async with AsyncJSONPlaceholderClient() as client:
            await client.bulk_create_users([{"name": "A"}], retry_non_idempotent=flag)

    assert mock_create.await_args.kwargs["retry_non_idempotent"] is flag


@respx.mock
async def test_async_health_check_success():
    route = respx.get(f"{BASE_URL}/users", params={"_limit": 1}).respond(
        json=[{"id": 1, "name": "User 1"}]
    )

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.health_check()

    assert result is True
    assert route.call_count == 1


@respx.mock
async def test_async_health_check_connection_error():
    """
    httpx.ConnectError が APIConnectionError に変換された後、
    health_check が False を返す契約を固定する。
    """
    route = respx.get(f"{BASE_URL}/users", params={"_limit": 1}).mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    async with AsyncJSONPlaceholderClient(retry_count=0) as client:
        result = await client.health_check()

    assert result is False
    assert route.call_count == 1


@respx.mock
async def test_async_health_check_log_structure() -> None:
    respx.get(f"{BASE_URL}/users", params={"_limit": 1}).mock(
        side_effect=httpx.ConnectError("Connection refused to secret-host.internal")
    )

    async with AsyncJSONPlaceholderClient(retry_count=0) as client:
        with patch.object(client, "logger") as mock_logger:
            result = await client.health_check()

    assert result is False

    assert mock_logger.warning.call_count == 2

    health_check_call = next(
        (c for c in mock_logger.warning.call_args_list if c[0][0] == "health_check_failed"),
        None,
    )
    assert health_check_call is not None, "health_check_failed ログが出力されていない"

    assert health_check_call[1]["error_type"] == "APIRetryError"
    assert health_check_call[1]["endpoint"] == "/users"
    # セキュリティ: error フィールド省略（_classify_error と同方針）
    assert "error" not in health_check_call[1]
    # async_all_retries_failed の error フィールド省略検証（機密情報保護）
    all_retries_call = next(
        (c for c in mock_logger.error.call_args_list if c[0][0] == "async_all_retries_failed"),
        None,
    )
    assert all_retries_call is not None, "async_all_retries_failed ログが出力されていること"
    assert "error" not in all_retries_call[1]


@respx.mock
async def test_get_user_data_parallel_requests():
    """
    TaskGroup の4 API呼び出しと posts の userId フィルタを通る統合データ契約を固定する。
    """
    user_id = 1

    route_user = respx.get(f"{BASE_URL}/users/{user_id}").respond(json=make_canonical_user(user_id))

    route_posts = respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(
        json=[
            {"id": 1, "userId": 1, "title": "Post by User 1", "body": "Content 1"},
            {"id": 3, "userId": 1, "title": "Another post by User 1", "body": "Content 3"},
        ]
    )

    route_todos = respx.get(f"{BASE_URL}/todos", params={"userId": user_id}).respond(
        json=[
            {"id": 1, "userId": 1, "title": "Todo 1", "completed": True},
            {"id": 2, "userId": 1, "title": "Todo 2", "completed": False},
        ]
    )

    route_albums = respx.get(f"{BASE_URL}/albums", params={"userId": user_id}).respond(
        json=[
            {"id": 1, "userId": 1, "title": "Album 1"},
            {"id": 2, "userId": 1, "title": "Album 2"},
        ]
    )

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_user_data(user_id)

    assert isinstance(result["user"], User)

    assert result["user"].id == 1
    assert result["user"].name == "Leanne Graham"

    assert len(result["posts"]) == 2
    assert all(isinstance(post, Post) for post in result["posts"])
    assert all(post.user_id == 1 for post in result["posts"])
    assert result["posts"][0].id == 1
    assert result["posts"][1].id == 3

    assert len(result["todos"]) == 2
    assert all(isinstance(todo, Todo) for todo in result["todos"])
    assert all(todo.user_id == 1 for todo in result["todos"])

    assert len(result["albums"]) == 2
    assert all(isinstance(album, Album) for album in result["albums"])
    assert all(album.user_id == 1 for album in result["albums"])

    assert route_user.call_count == 1
    assert route_posts.call_count == 1
    assert route_todos.call_count == 1
    assert route_albums.call_count == 1


@respx.mock
async def test_get_user_data_with_empty_posts():
    user_id = 1

    route_user = respx.get(f"{BASE_URL}/users/{user_id}").respond(
        json={
            "id": 1,
            "name": "Leanne Graham",
            "username": "Bret",
            "email": "Sincere@april.biz",
            "address": {
                "street": "Kulas Light",
                "suite": "Apt. 556",
                "city": "Gwenborough",
                "zipcode": "92998-3874",
                "geo": {"lat": "-37.3159", "lng": "81.1496"},
            },
            "phone": "1-770-736-8031 x56442",
            "website": "https://hildegard.org",
            "company": {
                "name": "Romaguera-Crona",
                "catchPhrase": "Multi-layered client-server neural-net",
                "bs": "harness real-time e-markets",
            },
        }
    )

    route_posts = respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(json=[])

    route_todos = respx.get(f"{BASE_URL}/todos", params={"userId": user_id}).respond(
        json=[{"id": 1, "userId": 1, "title": "Todo 1", "completed": True}]
    )

    route_albums = respx.get(f"{BASE_URL}/albums", params={"userId": user_id}).respond(
        json=[{"id": 1, "userId": 1, "title": "Album 1"}]
    )

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_user_data(user_id)

    assert result["posts"] == []
    assert len(result["todos"]) == 1
    assert len(result["albums"]) == 1
    assert isinstance(result["user"], User)
    assert all(isinstance(todo, Todo) for todo in result["todos"])
    assert all(isinstance(album, Album) for album in result["albums"])

    assert route_user.call_count == 1
    assert route_posts.call_count == 1
    assert route_todos.call_count == 1
    assert route_albums.call_count == 1


@respx.mock
async def test_get_user_data_user_not_found():
    """
    TaskGroup の ExceptionGroup から APIHTTPError を unwrap し、
    単一例外として伝播する契約を固定する。
    """
    user_id = 999

    route_user = respx.get(f"{BASE_URL}/users/{user_id}").respond(status_code=404)

    # TaskGroup は 1 タスク失敗時に残りを自動キャンセルするため、兄弟タスクは
    # キャンセルのタイミング次第で呼ばれない場合がある（call_count <= 1）。
    route_posts = respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(json=[])
    route_todos = respx.get(f"{BASE_URL}/todos", params={"userId": user_id}).respond(json=[])
    route_albums = respx.get(f"{BASE_URL}/albums", params={"userId": user_id}).respond(json=[])

    # retry_count=0 でリトライなし（即時失敗）
    async with AsyncJSONPlaceholderClient(retry_count=0) as client:
        with pytest.raises(APIHTTPError) as exc_info:
            await client.get_user_data(user_id)

    assert exc_info.value.status_code == 404

    assert route_user.call_count == 1
    assert route_posts.call_count <= 1
    assert route_todos.call_count <= 1
    assert route_albums.call_count <= 1


@respx.mock
async def test_get_user_data_posts_server_error():
    """
    TaskGroup の ExceptionGroup から APIRetryError を unwrap し、
    単一例外として伝播する契約を固定する。
    """
    user_id = 1

    route_user = respx.get(f"{BASE_URL}/users/{user_id}").respond(
        json={
            "id": 1,
            "name": "Leanne Graham",
            "username": "Bret",
            "email": "Sincere@april.biz",
            "address": {
                "street": "Kulas Light",
                "suite": "Apt. 556",
                "city": "Gwenborough",
                "zipcode": "92998-3874",
                "geo": {"lat": "-37.3159", "lng": "81.1496"},
            },
            "phone": "1-770-736-8031 x56442",
            "website": "https://hildegard.org",
            "company": {
                "name": "Romaguera-Crona",
                "catchPhrase": "Multi-layered client-server neural-net",
                "bs": "harness real-time e-markets",
            },
        }
    )

    route_posts = respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(
        status_code=500
    )

    route_todos = respx.get(f"{BASE_URL}/todos", params={"userId": user_id}).respond(json=[])
    route_albums = respx.get(f"{BASE_URL}/albums", params={"userId": user_id}).respond(json=[])

    # retry_count=0 でリトライなし（即時失敗）
    async with AsyncJSONPlaceholderClient(retry_count=0) as client:
        with pytest.raises(APIRetryError):
            await client.get_user_data(user_id)

    # 失敗タスク（posts）は必ず1回呼ばれる（retry_count=0 のためリトライなし）。
    # 兄弟タスク（user/todos/albums）は TaskGroup のキャンセルにより呼ばれない

    assert route_posts.call_count == 1
    assert route_user.call_count <= 1
    assert route_todos.call_count <= 1
    assert route_albums.call_count <= 1


async def test_get_user_data_multiple_failures_raises_single_api_client_error() -> None:
    """
    複数失敗でも ExceptionGroup を漏らさず、
    最初の APIClientError 系例外へ unwrap する契約を固定する。

    asyncio.sleep(0) は複数失敗を同時成立させるための同期バリア。
    """
    user_id = 1

    async def failing_user(*_: object, **__: object) -> User:
        await asyncio.sleep(0)
        raise APIHTTPError("User not found", status_code=404)

    async def failing_posts(*_: object, **__: object) -> list[Post]:
        await asyncio.sleep(0)
        raise APIRetryError("Retry limit exceeded")

    async def succeeding_todos(*_: object, **__: object) -> list[Todo]:
        await asyncio.sleep(0)
        return []

    async def succeeding_albums(*_: object, **__: object) -> list[Album]:
        await asyncio.sleep(0)
        return []

    async with AsyncJSONPlaceholderClient() as client:
        with (
            patch.object(client, "get_user", failing_user),
            patch.object(client, "get_posts", failing_posts),
            patch.object(client, "get_todos", succeeding_todos),
            patch.object(client, "get_albums", succeeding_albums),
        ):
            with pytest.raises(APIClientError) as exc_info:
                await client.get_user_data(user_id)

    assert not isinstance(exc_info.value, BaseExceptionGroup)

    assert isinstance(exc_info.value, APIHTTPError | APIRetryError)


async def test_get_user_data_single_failure_cancels_pending_sibling_task() -> None:
    """
    TaskGroup の fail-fast により未完了の兄弟タスクをキャンセルし、待機を短絡する契約を固定する。
    """
    user_id = 1
    todos_started = asyncio.Event()

    async def failing_user(*_: object, **__: object) -> User:
        raise APIHTTPError("User not found", status_code=404)

    async def slow_todos(*_: object, **__: object) -> list[Todo]:
        todos_started.set()
        await asyncio.sleep(5)  # fail-fastでキャンセルされる想定の長時間sleep
        return []

    async with AsyncJSONPlaceholderClient() as client:
        with (
            patch.object(client, "get_user", failing_user),
            patch.object(client, "get_todos", slow_todos),
            patch.object(client, "get_posts", AsyncMock(return_value=[])),
            patch.object(client, "get_albums", AsyncMock(return_value=[])),
        ):
            start = time.monotonic()
            with pytest.raises(APIHTTPError):
                await asyncio.wait_for(client.get_user_data(user_id), timeout=1.0)
            elapsed = time.monotonic() - start

    assert todos_started.is_set()

    assert elapsed < 1.0


@pytest.mark.parametrize(
    "fatal_exc",
    [MemoryError("OOM"), RecursionError("max depth")],
)
async def test_get_user_data_fatal_exception_not_wrapped_as_api_client_error(
    fatal_exc: BaseException,
) -> None:
    """
    ASYNC_FATAL_EXCEPTIONS は APIClientError に誤変換せず、
    TaskGroup の ExceptionGroup として伝播する契約を固定する。
    """
    user_id = 1

    async def failing_user(*_: object, **__: object) -> User:
        raise fatal_exc

    async with AsyncJSONPlaceholderClient() as client:
        with (
            patch.object(client, "get_user", failing_user),
            patch.object(client, "get_posts", AsyncMock(return_value=[])),
            patch.object(client, "get_todos", AsyncMock(return_value=[])),
            patch.object(client, "get_albums", AsyncMock(return_value=[])),
        ):
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await client.get_user_data(user_id)

    assert not isinstance(exc_info.value, APIClientError)
    assert any(isinstance(e, type(fatal_exc)) for e in exc_info.value.exceptions)


@pytest.mark.parametrize("invalid_user_id", [0, -1], ids=["zero", "negative"])
@respx.mock
async def test_get_user_data_invalid_user_id_raises_bare_value_error(
    invalid_user_id: int,
) -> None:
    """
    validation 前に get_user がネットワークへ出るため、
    respx.mock で実行経路を閉じた上で ValueError の unwrap 契約を検証する。
    """
    respx.get(f"{BASE_URL}/users/{invalid_user_id}").respond(json=make_canonical_user(1))

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match="user_id must be >= 1") as exc_info:
            await client.get_user_data(invalid_user_id)

    assert not isinstance(exc_info.value, BaseExceptionGroup)


@pytest.mark.parametrize(
    "limit,expected_count,test_description",
    [
        (2, 2, "limit=2で2件取得"),
        (None, 5, "limit=Noneで全件取得（モックは5件）"),
        (0, 0, "limit=0で0件取得（境界値検証、API仕様では空配列返却）"),
        (100, 5, "limit=100で上限超過時は全件取得"),
    ],
    ids=["normal_limit", "no_limit", "zero_limit", "excessive_limit"],
)
@respx.mock
async def test_get_posts_with_various_limits(limit, expected_count, test_description):

    all_posts = [
        {"id": i, "userId": 1, "title": f"Post {i}", "body": f"Content {i}"} for i in range(1, 6)
    ]

    if limit is None:
        respx.get(f"{BASE_URL}/posts").respond(json=all_posts)
        expected_posts = all_posts
    elif limit == 0:
        respx.get(f"{BASE_URL}/posts", params={"_limit": 0}).respond(json=[])
        expected_posts = []
    else:
        respx.get(f"{BASE_URL}/posts", params={"_limit": limit}).respond(json=all_posts[:limit])
        expected_posts = all_posts[:limit]

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_posts(limit=limit)

    actual_expected = len(expected_posts)
    assert len(result) == actual_expected, (
        f"{test_description}: expected {actual_expected}, got {len(result)}"
    )
    assert [post.model_dump(by_alias=True) for post in result] == expected_posts
    assert all(isinstance(post, Post) for post in result)


@pytest.mark.parametrize(
    "user_id,expected_count,test_description",
    [
        (None, 5, "user_id=Noneで全投稿取得（フィルタなし）"),
        (1, 2, "user_id=1でユーザー1の投稿のみ取得（API側フィルタ）"),
        (2, 1, "user_id=2でユーザー2の投稿のみ取得"),
        (999, 0, "user_id=999で存在しないユーザー（空配列返却）"),
    ],
    ids=["no_filter", "user_1", "user_2", "nonexistent_user"],
)
@respx.mock
async def test_async_get_posts_user_filter(user_id, expected_count, test_description):

    all_posts = [
        {"id": 1, "userId": 1, "title": "Post 1 by User 1", "body": "Content 1"},
        {"id": 2, "userId": 2, "title": "Post 2 by User 2", "body": "Content 2"},
        {"id": 3, "userId": 1, "title": "Post 3 by User 1", "body": "Content 3"},
        {"id": 4, "userId": 3, "title": "Post 4 by User 3", "body": "Content 4"},
        {"id": 5, "userId": 3, "title": "Post 5 by User 3", "body": "Content 5"},
    ]

    if user_id is None:
        respx.get(f"{BASE_URL}/posts").respond(json=all_posts)
        expected_posts = all_posts
    elif user_id == 999:
        respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(json=[])
        expected_posts = []
    else:
        filtered_posts = [p for p in all_posts if p["userId"] == user_id]
        respx.get(f"{BASE_URL}/posts", params={"userId": user_id}).respond(json=filtered_posts)
        expected_posts = filtered_posts

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_posts(user_id=user_id)

    assert len(result) == expected_count, (
        f"{test_description}: expected {expected_count}, got {len(result)}"
    )
    assert [post.model_dump(by_alias=True) for post in result] == expected_posts
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
async def test_async_get_posts_validation_error(limit, user_id, expected_error):
    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            await client.get_posts(limit=limit, user_id=user_id)


@respx.mock
async def test_get_post_by_id_success():
    post_id = 1
    expected_post = {"id": 1, "userId": 1, "title": "Test Post", "body": "Test Content"}

    route = respx.get(f"{BASE_URL}/posts/{post_id}").respond(json=expected_post)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_post(post_id)

    assert result.model_dump(by_alias=True) == expected_post
    assert result.id == post_id
    assert result.title is not None
    assert result.body is not None
    assert route.call_count == 1


@respx.mock
async def test_get_post_not_found():
    post_id = 999999

    respx.get(f"{BASE_URL}/posts/{post_id}").respond(status_code=404)

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(APIHTTPError) as exc_info:
            await client.get_post(post_id)

        assert exc_info.value.status_code == 404


@respx.mock
async def test_create_post_success():
    title = "New Post"
    body = "This is a new post content"
    user_id = 1

    expected_response = {
        "id": 101,
        "userId": user_id,
        "title": title,
        "body": body,
    }

    route = respx.post(f"{BASE_URL}/posts").respond(status_code=201, json=expected_response)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.create_post(title=title, body=body, user_id=user_id)

    assert result.id == 101
    assert result.user_id == user_id
    assert result.title == title
    assert result.body == body

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["title"] == title
    assert request_body["body"] == body
    assert request_body["userId"] == user_id


@respx.mock
async def test_create_post_with_empty_body():
    title = "Post with Empty Body"
    body = ""
    user_id = 1

    expected_response = {"id": 102, "userId": user_id, "title": title, "body": body}

    route = respx.post(f"{BASE_URL}/posts").respond(status_code=201, json=expected_response)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.create_post(title=title, body=body, user_id=user_id)

    assert result.id == 102
    assert result.body == ""

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["body"] == ""
    assert request_body["title"] == title
    assert request_body["userId"] == user_id


@pytest.mark.parametrize(
    "user_id,completed,limit,expected_params,expected_count,test_description",
    [
        (1, True, 5, {"userId": 1, "completed": True, "_limit": 5}, 2, "全パラメータ指定"),
        (1, None, None, {"userId": 1}, 3, "user_idのみ指定"),
        (None, False, 10, {"completed": False, "_limit": 10}, 2, "completedとlimit指定"),
        (None, None, None, {}, 5, "パラメータなし（全取得）"),
        (2, True, None, {"userId": 2, "completed": True}, 1, "user_id=2, completed=True"),
    ],
    ids=[
        "all_params",
        "user_id_only",
        "completed_and_limit",
        "no_params",
        "user2_completed",
    ],
)
@respx.mock
async def test_get_todos_with_filters(
    user_id, completed, limit, expected_params, expected_count, test_description
):

    all_todos = [
        {"id": 1, "userId": 1, "title": "Todo 1", "completed": True},
        {"id": 2, "userId": 1, "title": "Todo 2", "completed": False},
        {"id": 3, "userId": 1, "title": "Todo 3", "completed": True},
        {"id": 4, "userId": 2, "title": "Todo 4", "completed": True},
        {"id": 5, "userId": 2, "title": "Todo 5", "completed": False},
    ]

    filtered_todos = all_todos
    if user_id is not None:
        filtered_todos = [t for t in filtered_todos if t["userId"] == user_id]
    if completed is not None:
        filtered_todos = [t for t in filtered_todos if t["completed"] == completed]
    if limit is not None:
        filtered_todos = filtered_todos[:limit]

    route = respx.get(f"{BASE_URL}/todos", params__eq=expected_params).respond(json=filtered_todos)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_todos(user_id=user_id, completed=completed, limit=limit)

    assert len(result) == expected_count, (
        f"{test_description}: expected {expected_count}, got {len(result)}"
    )
    assert [todo.model_dump(by_alias=True) for todo in result] == filtered_todos
    assert all(isinstance(todo, Todo) for todo in result)
    assert route.call_count == 1

    if user_id is not None:
        assert all(t.user_id == user_id for t in result)
    if completed is not None:
        assert all(t.completed == completed for t in result)


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
async def test_async_get_todos_validation_error(limit, user_id, expected_error):
    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            await client.get_todos(limit=limit, user_id=user_id)


@respx.mock
async def test_async_get_todo() -> None:
    """sync 側 test_sync_get_todo と対を成す。"""
    mock_todo = {"id": 1, "userId": 1, "title": "delectus aut autem", "completed": False}

    route = respx.get(f"{BASE_URL}/todos/1").respond(json=mock_todo)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_todo(1)

    assert result.id == 1
    assert result.title == "delectus aut autem"
    assert result.completed is False
    assert route.call_count == 1


@respx.mock
async def test_async_create_todo() -> None:
    """sync 側 test_sync_create_todo と対を成す。"""
    new_todo_response = {
        "id": 201,
        "title": "Buy groceries",
        "userId": 1,
        "completed": False,
    }

    route = respx.post(f"{BASE_URL}/todos").respond(status_code=201, json=new_todo_response)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.create_todo(title="Buy groceries", user_id=1, completed=False)

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
    "user_id,expected_count,test_description",
    [
        (1, 2, "user_id=1でアルバム取得"),
        (None, 5, "user_id指定なしで全アルバム取得"),
        (2, 1, "user_id=2でアルバム取得"),
    ],
    ids=["user_id_1", "no_user_id", "user_id_2"],
)
@respx.mock
async def test_get_albums_with_filters(user_id, expected_count, test_description):

    all_albums = [
        {"id": 1, "userId": 1, "title": "Album 1"},
        {"id": 2, "userId": 1, "title": "Album 2"},
        {"id": 3, "userId": 2, "title": "Album 3"},
        {"id": 4, "userId": 3, "title": "Album 4"},
        {"id": 5, "userId": 3, "title": "Album 5"},
    ]

    filtered_albums = (
        [a for a in all_albums if a["userId"] == user_id] if user_id is not None else all_albums
    )
    expected_params = {"userId": user_id} if user_id is not None else {}
    route = respx.get(f"{BASE_URL}/albums", params__eq=expected_params).respond(
        json=filtered_albums
    )

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_albums(user_id=user_id)

    assert len(result) == expected_count, (
        f"{test_description}: expected {expected_count}, got {len(result)}"
    )
    assert [album.model_dump(by_alias=True) for album in result] == filtered_albums
    assert all(isinstance(album, Album) for album in result)
    assert route.call_count == 1

    if user_id is not None:
        assert all(a.user_id == user_id for a in result)


@pytest.mark.parametrize(
    "user_id,expected_error",
    [
        (0, "user_id must be >= 1"),
        (-1, "user_id must be >= 1"),
    ],
    ids=["zero_user_id", "negative_user_id"],
)
async def test_async_get_albums_validation_error(user_id, expected_error):
    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match=expected_error):
            await client.get_albums(user_id=user_id)


@pytest.mark.parametrize(
    "album_id,expected_count,test_description",
    [
        (1, 2, "album_id=1で写真取得"),
        (None, 6, "album_id指定なしで全写真取得"),
        (2, 1, "album_id=2で写真取得"),
    ],
    ids=["album_id_1", "no_album_id", "album_id_2"],
)
@respx.mock
async def test_get_photos_with_filters(album_id, expected_count, test_description):

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

    if album_id is not None:
        filtered_photos = [p for p in all_photos if p["albumId"] == album_id]
        url = f"{BASE_URL}/albums/{album_id}/photos"
    else:
        filtered_photos = all_photos
        url = f"{BASE_URL}/photos"

    respx.get(url).respond(json=filtered_photos)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_photos(album_id=album_id)

    assert len(result) == expected_count, (
        f"{test_description}: expected {expected_count}, got {len(result)}"
    )
    assert [photo.model_dump(by_alias=True) for photo in result] == filtered_photos
    assert all(isinstance(photo, Photo) for photo in result)

    if album_id is not None:
        assert all(p.album_id == album_id for p in result)


@respx.mock
async def test_async_get_comments_with_post_id() -> None:
    mock_comments = [
        {"id": 1, "postId": 1, "name": "Test Comment", "email": "test@example.com", "body": "Body"},
    ]
    route = respx.get(f"{BASE_URL}/posts/1/comments").respond(json=mock_comments)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_comments(post_id=1)

    assert route.call_count == 1
    assert [comment.model_dump(by_alias=True) for comment in result] == mock_comments


@respx.mock
async def test_async_get_comments_without_post_id() -> None:
    mock_comments = [
        {"id": 1, "postId": 1, "name": "Comment 1", "email": "a@b.com", "body": "Body 1"},
        {"id": 2, "postId": 2, "name": "Comment 2", "email": "c@d.com", "body": "Body 2"},
    ]
    route = respx.get(f"{BASE_URL}/comments", params__eq={}).respond(json=mock_comments)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_comments()

    assert route.call_count == 1
    assert [comment.model_dump(by_alias=True) for comment in result] == mock_comments


@pytest.mark.parametrize(
    "post_id",
    [0, -1, -100],
    ids=["post_id_zero", "post_id_negative", "post_id_large_negative"],
)
async def test_async_get_comments_invalid_post_id(
    post_id: int,
) -> None:
    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match="post_id must be >= 1"):
            await client.get_comments(post_id=post_id)


@pytest.mark.parametrize(
    "album_id",
    [0, -1, -100],
    ids=["album_id_zero", "album_id_negative", "album_id_large_negative"],
)
async def test_async_get_photos_invalid_album_id(album_id: int) -> None:
    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(ValueError, match="album_id must be >= 1"):
            await client.get_photos(album_id=album_id)


@respx.mock
async def test_async_get_users(sample_users_list: list[dict[str, Any]]) -> None:
    route = respx.get(f"{BASE_URL}/users").respond(json=sample_users_list[:2])

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_users()

    assert len(result) == 2
    assert result[0].name == "Leanne Graham"
    assert result[1].id == 2
    assert route.call_count == 1


@respx.mock
async def test_async_get_users_returns_empty_list() -> None:
    route = respx.get(f"{BASE_URL}/users").respond(json=[])

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.get_users()

    assert result == []
    assert route.call_count == 1


@respx.mock
async def test_async_update_todo() -> None:
    todo_id = 1
    patch_data = {"completed": True}
    full_response = {"id": todo_id, "title": "delectus aut autem", "completed": True, "userId": 1}

    route = respx.patch(f"{BASE_URL}/todos/{todo_id}").respond(status_code=200, json=full_response)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.update_todo(todo_id, completed=True)

    assert result == full_response
    assert result["completed"] is True
    assert route.call_count == 1
    assert route.calls[0].request.method == "PATCH"

    request_body = json.loads(route.calls[0].request.content)
    assert request_body == patch_data


@respx.mock
async def test_async_update_todo_multiple_fields() -> None:
    todo_id = 1
    full_response = {
        "id": todo_id,
        "title": "Updated Title",
        "completed": True,
        "userId": 1,
    }

    route = respx.patch(f"{BASE_URL}/todos/{todo_id}").respond(status_code=200, json=full_response)

    async with AsyncJSONPlaceholderClient() as client:
        result = await client.update_todo(todo_id, title="Updated Title", completed=True)

    assert result["title"] == "Updated Title"
    assert result["completed"] is True
    assert route.call_count == 1

    request_body = json.loads(route.calls[0].request.content)
    assert request_body == {"title": "Updated Title", "completed": True}


async def test_bulk_create_users_details_truncated_false_at_max() -> None:
    client = AsyncJSONPlaceholderClient(base_url="https://test.com")
    with patch.object(client, "create_user", new=AsyncMock(side_effect=RuntimeError("fail"))):
        with capture_logs() as logs:
            result = await client.bulk_create_users(
                [{"name": f"u{i}"} for i in range(MAX_LOGGED_FAILURE_DETAILS)]
            )
    assert result == []
    warn = next((lg for lg in logs if lg.get("event") == "bulk_create_partial_failure"), None)
    assert warn is not None
    assert warn["failed_count"] == MAX_LOGGED_FAILURE_DETAILS
    assert warn["success_count"] == 0
    assert warn["details_truncated"] is False
    assert len(warn["failed_details"]) == MAX_LOGGED_FAILURE_DETAILS
    detail = warn["failed_details"][0]
    assert "index" in detail
    assert "error_type" in detail


async def test_bulk_create_users_details_truncated_true_above_max() -> None:
    client = AsyncJSONPlaceholderClient(base_url="https://test.com")
    with patch.object(client, "create_user", new=AsyncMock(side_effect=RuntimeError("fail"))):
        with capture_logs() as logs:
            result = await client.bulk_create_users(
                [{"name": f"u{i}"} for i in range(MAX_LOGGED_FAILURE_DETAILS + 1)]
            )
    assert result == []
    warn = next((lg for lg in logs if lg.get("event") == "bulk_create_partial_failure"), None)
    assert warn is not None
    assert warn["failed_count"] == MAX_LOGGED_FAILURE_DETAILS + 1
    assert warn["success_count"] == 0
    assert warn["details_truncated"] is True
    assert len(warn["failed_details"]) == MAX_LOGGED_FAILURE_DETAILS
    detail = warn["failed_details"][0]
    assert "index" in detail
    assert "error_type" in detail


@pytest.mark.parametrize(
    ("method_name", "http_verb", "path", "call_args"),
    [
        ("update_post", "put", "/posts/1", (1, "Title", "Body")),
        ("create_user", "post", "/users", ({"name": "New User"},)),
        ("update_todo", "patch", "/todos/1", (1,)),
    ],
)
@respx.mock
async def test_async_dict_returning_methods_reject_non_object_json(
    method_name: str,
    http_verb: str,
    path: str,
    call_args: tuple[object, ...],
) -> None:
    """sync 版と同一の契約違反検出を async 側でも保証する（parity）。"""
    route = getattr(respx, http_verb)(f"{BASE_URL}{path}").respond(
        status_code=200,
        json=["unexpected", "array"],
    )

    async with AsyncJSONPlaceholderClient() as client:
        with pytest.raises(APIJSONDecodeError) as exc_info:
            await getattr(client, method_name)(*call_args)

    assert "Expected object JSON response" in str(exc_info.value)
    assert "list" in str(exc_info.value)
    assert route.call_count == 1
