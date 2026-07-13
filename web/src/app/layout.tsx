import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Luis Benvenuto — Papers",
  description: "Publicaciones científicas — sincronizadas automáticamente desde GitHub.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="min-h-screen antialiased font-sans">{children}</body>
    </html>
  );
}
