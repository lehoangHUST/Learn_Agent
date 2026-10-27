"""
Lesson4_1: Tạo thêm các cấu trúc node
"""
import json
from typing import Iterable, Literal
from datetime import date
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from langchain_core.tools import StructuredTool

NewsCategory = Literal["kinh_te", "cong_nghe", "the_thao"]


class SearchNewsInput(BaseModel):
    """Validate arguments for searching news."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=3,
        max_length=200,
        description="Từ khóa từ 3 đến 200 ký tự để tìm trong kho bài báo.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Số kết quả tối đa, phải nằm trong khoảng từ 1 đến 10.",
    )
    category: NewsCategory | None = Field(
        default=None,
        description=(
            "Danh mục tùy chọn: kinh_te, cong_nghe hoặc the_thao. "
            "Bỏ trống nếu không muốn lọc theo danh mục."
        ),
    )
    start_date: date | None = Field(
        default=None,
        description="Ngày bắt đầu lọc, định dạng YYYY-MM-DD, có tính cả ngày này.",
    )
    end_date: date | None = Field(
        default=None,
        description="Ngày kết thúc lọc, định dạng YYYY-MM-DD, có tính cả ngày này.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query không được chỉ chứa khoảng trắng")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> "SearchNewsInput":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date phải nhỏ hơn hoặc bằng end_date")
        return self


class GetArticleInput(BaseModel):
    """Validate arguments for retrieving one article by ID."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    article_id: str = Field(
        pattern=r"^news-\d{3,}$",
        description="ID bài báo chính xác, ví dụ news-001 hoặc news-1234.",
    )


class LatestNewsInput(BaseModel):
    """Validate arguments for getting recently published news."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Số bài báo mới nhất cần lấy, từ 1 đến 10.",
    )
    category: NewsCategory | None = Field(
        default=None,
        description="Danh mục tùy chọn: kinh_te, cong_nghe hoặc the_thao.",
    )


class Article(BaseModel):
    
    """A normalized news article record."""
    
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    category: str = Field(min_length=1)
    published_at: date
    source: str = Field(default="Sample News", min_length=1)

    @property
    def searchable_text(self) -> str:
        """Text used by the demo keyword search."""
        return f"{self.title} {self.content} {self.source}".casefold()


class NewsRepository:
    
    """Small in-memory repository that can later be replaced by Elasticsearch.
    
    The tools depend on this interface rather than a global database client.
    """

    def __init__(self, articles: Iterable[Article]) -> None:
        # Copy records to avoid callers mutating the repository's collection.
        self._articles: tuple[Article, ...] = tuple(articles)

    def search(
        self,
        *,
        query: str,
        limit: int,
        category: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Article]:
        """Search title/content/source with simple case-insensitive substring matching.

        This is intentionally a demo implementation, not BM25 or semantic search.
        """
        query_normalized = query.casefold().strip()
        matches: list[Article] = []

        for article in self._articles:
            if query_normalized not in article.searchable_text:
                continue
            if category is not None and article.category != category:
                continue
            if start_date is not None and article.published_at < start_date:
                continue
            if end_date is not None and article.published_at > end_date:
                continue
            matches.append(article)

        matches.sort(key=lambda item: item.published_at, reverse=True)
        return matches[:limit]

    def get_by_id(self, article_id: str) -> Article | None:
        """Return the exact article matching ``article_id``."""
        for article in self._articles:
            if article.id == article_id:
                return article
        return None

    def latest(
        self,
        *,
        limit: int,
        category: str | None = None,
    ) -> list[Article]:
        """Return newest articles, optionally filtered by category."""
        articles = [
            article
            for article in self._articles
            if category is None or article.category == category
        ]
        articles.sort(key=lambda item: item.published_at, reverse=True)
        return articles[:limit]

def build_sample_repository() -> NewsRepository:
    
    return NewsRepository(
        [
            Article(
                id="news-001",
                title="VN-Index tăng điểm nhờ nhóm ngân hàng",
                content=(
                    "Thị trường chứng khoán Việt Nam tăng điểm trong phiên giao dịch. "
                    "Nhóm ngân hàng và cổ phiếu vốn hóa lớn đóng góp tích cực cho VN-Index."
                ),
                category="kinh_te",
                published_at=date(2026, 10, 25),
                source="VNews Demo",
            ),
            Article(
                id="news-002",
                title="Doanh nghiệp công nghệ tăng đầu tư vào AI",
                content=(
                    "Nhiều doanh nghiệp công nghệ mở rộng ngân sách trí tuệ nhân tạo, "
                    "hạ tầng dữ liệu và tự động hóa quy trình kinh doanh."
                ),
                category="cong_nghe",
                published_at=date(2026, 10, 24),
                source="VNews Demo",
            ),
            Article(
                id="news-003",
                title="Thanh khoản thị trường chứng khoán cải thiện",
                content=(
                    "Thanh khoản chứng khoán tăng so với phiên trước, trong khi nhà đầu tư "
                    "theo dõi diễn biến lãi suất và kết quả kinh doanh quý."
                ),
                category="kinh_te",
                published_at=date(2026, 10, 23),
                source="VNews Demo",
            ),
            Article(
                id="news-004",
                title="Đội tuyển bóng đá chuẩn bị cho giải đấu mới",
                content=(
                    "Đội tuyển bóng đá tăng cường tập luyện và thử nghiệm chiến thuật "
                    "trước khi bước vào giải đấu tiếp theo."
                ),
                category="the_thao",
                published_at=date(2026, 10, 22),
                source="VNews Demo",
            ),
            Article(
                id="news-005",
                title="Ngân hàng số cải tiến trải nghiệm khách hàng",
                content=(
                    "Một số ngân hàng công bố tính năng số mới, tăng cường xác thực "
                    "và cá nhân hóa dịch vụ trên ứng dụng di động."
                ),
                category="cong_nghe",
                published_at=date(2026, 10, 21),
                source="VNews Demo",
            ),
        ]
    )
    
def _format_article_card(article: Article) -> str:
    """Format a compact record suitable for an LLM context window."""
    excerpt = article.content.strip()
    if len(excerpt) > 220:
        excerpt = excerpt[:217].rstrip() + "..."

    return (
        f"ID: {article.id}\n"
        f"Tiêu đề: {article.title}\n"
        f"Nguồn: {article.source}\n"
        f"Ngày đăng: {article.published_at.isoformat()}\n"
        f"Danh mục: {article.category}\n"
        f"Trích đoạn: {excerpt}"
    )
    
def build_news_tools(
    repository: NewsRepository,
) -> tuple[StructuredTool, StructuredTool, StructuredTool]:
    """Create tools bound to an explicit repository instance.

    No LLM is created here. The returned tools can be invoked independently,
    unit-tested, or passed to a LangChain agent later.
    """

    def _search_news(
        query: str,
        limit: int = 5,
        category: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> str:
        """Search the internal news repository by keyword and optional filters."""
        articles = repository.search(
            query=query,
            limit=limit,
            category=category,
            start_date=start_date,
            end_date=end_date,
        )
        if not articles:
            return "Không tìm thấy bài báo phù hợp với điều kiện tìm kiếm."
        return "\n\n---\n\n".join(_format_article_card(item) for item in articles)

    def _get_article_by_id(article_id: str) -> str:
        """Get a single article by its exact ID; do not use this for keyword search."""
        article = repository.get_by_id(article_id)
        if article is None:
            return f"Không tìm thấy bài báo có ID '{article_id}'."
        return (
            f"ID: {article.id}\n"
            f"Tiêu đề: {article.title}\n"
            f"Nguồn: {article.source}\n"
            f"Ngày đăng: {article.published_at.isoformat()}\n"
            f"Danh mục: {article.category}\n\n"
            f"Nội dung:\n{article.content}"
        )

    def _get_latest_news(
        limit: int = 5,
        category: str | None = None,
    ) -> str:
        """List the most recently published articles, optionally by category."""
        articles = repository.latest(limit=limit, category=category)
        if not articles:
            return "Chưa có bài báo phù hợp với điều kiện lọc."
        return "\n\n---\n\n".join(_format_article_card(item) for item in articles)

    search_news = StructuredTool.from_function(
        func=_search_news,
        name="search_news",
        description=(
            "Tìm bài báo trong kho tin nội bộ theo từ khóa, danh mục và khoảng thời gian. "
            "Dùng khi người dùng muốn khám phá một chủ đề hoặc tìm các bài liên quan. "
            "Không dùng tool này để lấy bài theo ID đã biết; dùng get_article_by_id. "
            "Đây là tìm kiếm từ khóa, không phải truy vấn web bên ngoài."
        ),
        args_schema=SearchNewsInput,
        infer_schema=False,
        handle_validation_error=False,
    )

    get_article_by_id = StructuredTool.from_function(
        func=_get_article_by_id,
        name="get_article_by_id",
        description=(
            "Lấy nội dung của một bài báo theo ID chính xác, ví dụ news-001. "
            "Dùng khi ID đã được biết từ kết quả tìm kiếm hoặc do ứng dụng cung cấp. "
            "Không dùng để tìm bài theo chủ đề hoặc từ khóa."
        ),
        args_schema=GetArticleInput,
        infer_schema=False,
        handle_validation_error=False,
    )

    get_latest_news = StructuredTool.from_function(
        func=_get_latest_news,
        name="get_latest_news",
        description=(
            "Lấy các bài báo mới nhất đã có trong kho dữ liệu nội bộ. "
            "Dùng khi người dùng hỏi tin mới nhất hoặc tin gần đây mà không cần "
            "truy vấn theo một từ khóa cụ thể. Có thể lọc theo danh mục. "
            "Không tuyên bố đây là tin mới nhất trên toàn Internet."
        ),
        args_schema=LatestNewsInput,
        infer_schema=False,
        handle_validation_error=False,
    )

    return search_news, get_article_by_id, get_latest_news

def main() -> None:
    
    repository = build_sample_repository()
    search_news_tool, get_article_by_id_tool, get_latest_news_tool = build_news_tools(repository)

    print("=" * 72)
    print("CUSTOM TOOLS LAB — direct invocation (no LLM required)")
    print("=" * 72)

    print("\n[1] Tool metadata")
    for item in (search_news_tool, get_article_by_id_tool, get_latest_news_tool):
        print(f"\nname: {item.name}")
        print(f"description: {item.description}")
        print("args:")
        print(json.dumps(item.args, ensure_ascii=False, indent=2))

    print("\n[2] search_news")
    print(
        search_news_tool.invoke(
            {
                "query": "chứng khoán",
                "limit": 3,
                "category": "kinh_te",
                "start_date": "2026-10-01",
                "end_date": "2026-10-26",
            }
        )
    )

    print("\n[3] get_article_by_id")
    print(get_article_by_id_tool.invoke({"article_id": "news-001"}))

    print("\n[4] get_latest_news")
    print(get_latest_news_tool.invoke({"limit": 2}))

    print("\n[5] Invalid input — expected validation error")
    try:
        search_news_tool.invoke({"query": "AI laf gif ?", "limit": 2})
    except Exception as exc:  # demo: display the framework's actual error type
        print(f"{type(exc).__name__}: {exc}")
    

if __name__ == "__main__":
    
    main()