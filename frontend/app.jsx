// src/App.tsx
import React, { useState } from 'react';
import { ChatMessage, ChatResponse } from './types';
import './src/styles.css';

const App: React.FC = () => {
  const [value, setValue] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const surpriseOptions = [
    'Macbook Air giá như thế nào?',
    'Các dòng laptop của Lenovo',
    'MSI modern có được giảm giá không?',
  ];

  const surprise = () => {
    const randomIndex = Math.floor(Math.random() * surpriseOptions.length);
    setValue(surpriseOptions[randomIndex]);
  };

  const getResponse = async () => {
    if (!value.trim()) {
      setError('Vui lòng nhập nội dung cần tìm kiếm!');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: value,
          history: chatHistory,
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data: ChatResponse = await response.json();

      setChatHistory((prev) => [
        ...prev,
        { role: 'user', parts: value },
        {
          role: 'assistant',
          parts: data.content,
          products: data.products
        },
      ]);

      setValue('');

    } catch (error) {
      console.error('Error:', error);
      setError('Đã xảy ra lỗi! Vui lòng thử lại sau.');
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setValue('');
    setError('');
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
          <button onClick={getResponse} disabled={loading}>
            {loading ? 'Đang xử lý...' : 'Tìm kiếm'}
          </button>
        )}
        {error && <button onClick={clear}>Xóa</button>}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="chat-history">
        {chatHistory.map((chatItem, index) => (
          <div
            key={index}
            className={`chat-item ${chatItem.role}`}
          >
            <p className="message">{chatItem.parts}</p>

            {chatItem.role === 'assistant' && chatItem.products && (
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