import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Luis Benvenuto — Papers",
  description: "Scientific publications — automatically synced from GitHub.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased font-sans">{children}</body>
    </html>
  );
}
