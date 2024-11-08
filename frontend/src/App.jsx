import { useState } from "react";
import "./styles.css";

const App = () => {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const surpriseOptions = [
    'Tìm cho tôi điện thoại Samsung tầm giá 10 triệu',
    'So sánh iPhone 14 Pro Max và Samsung Galaxy S23 Ultra',
    'Điện thoại nào có camera tốt nhất trong tầm giá 15 triệu?',
  ];

  const surprise = () => {
    const randomIndex = Math.floor(Math.random() * surpriseOptions.length);
    setValue(surpriseOptions[randomIndex]);
  };

  const getResponse = async () => {
    if (!value.trim()) {
      setError("Vui lòng nhập nội dung cần tìm kiếm!");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          history: chatHistory,
          message: value
        }),
      });

      if (!response.ok) {
        throw new Error('Có lỗi xảy ra khi kết nối với server');
      }

      const data = await response.json();

      setChatHistory(oldChatHistory => [
        ...oldChatHistory,
        {
          role: "user",
          parts: value
        },
        {
          role: "assistant",
          parts: data.content,
          products: data.products // Store product info from RAG
        }
      ]);

      setValue("");

    } catch (error) {
      console.error('Error:', error);
      setError("Đã xảy ra lỗi! Vui lòng thử lại sau.");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setValue("");
    setError("");
    setChatHistory([]);
  };

  return (
    <div className="app">
      <div className="header">
        <h1>Trợ lý mua sắm thông minh</h1>
        <button
          className="surprise"
          onClick={surprise}
          disabled={loading}
        >
          Gợi ý câu hỏi
        </button>
      </div>

      <div className="search-input">
        <input
          value={value}
          placeholder="Tìm kiếm sản phẩm, so sánh giá, hoặc hỏi thông tin chi tiết..."
          onChange={(e) => setValue(e.target.value)}
          disabled={loading}
        />
        {!error && (
          <button
            onClick={getResponse}
            disabled={loading}
            className="search-button"
          >
            {loading ? 'Đang xử lý...' : 'Tìm kiếm'}
          </button>
        )}
        {error && (
          <button onClick={clear} className="clear-button">
            Xóa
          </button>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="chat-history">
        {chatHistory.map((chatItem, index) => (
          <div
            key={index}
            className={`chat-item ${chatItem.role}`}
          >
            <p className="message">{chatItem.parts}</p>

            {/* Display products if available in assistant response */}
            {chatItem.role === "assistant" && chatItem.products && (
              <div className="products-grid">
                {chatItem.products.map((product, productIndex) => (
                  <div key={productIndex} className="product-card">
                    {product.image && (
                      <img
                        src={product.image}
                        alt={product.name}
                        className="product-image"
                      />
                    )}
                    <h3>{product.name}</h3>
                    <p className="price">
                      {product.price?.toLocaleString('vi-VN')}đ
                    </p>
                    {product.description && (
                      <p className="description">
                        {product.description.slice(0, 100)}...
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {loading && (
        <div className="loading">
          Đang xử lý câu hỏi của bạn...
        </div>
      )}
    </div>
  );
};

export default App;