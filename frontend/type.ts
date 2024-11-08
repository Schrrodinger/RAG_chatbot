// src/types.ts

export interface ChatMessage {
  role: string;
  parts: string;
  products?: Product[];
}

export interface Product {
  id: string;
  name: string;
  price: number;
  description?: string;
  image?: string;
}

export interface ChatResponse {
  role: string;
  content: string;
  products?: Product[];
  query?: string;
}