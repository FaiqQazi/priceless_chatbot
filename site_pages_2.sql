ALTER TABLE site_pages
ADD COLUMN deadline_date TEXT NOT NULL DEFAULT 'Unknown',
ADD COLUMN event_dates TEXT NOT NULL DEFAULT 'No specific event date',
ADD COLUMN discount TEXT NOT NULL DEFAULT 'No discount';
