-- GDP AI Platform Database Initialization (Continued)

-- Insert sample countries data (continued)
INSERT INTO countries (code, name, region, income_group, currency_code, population, capital) VALUES
('USA', 'United States', 'North America', 'High income', 'USD', 331000000, 'Washington D.C.'),
('CHN', 'China', 'East Asia & Pacific', 'Upper middle income', 'CNY', 1439000000, 'Beijing'),
('JPN', 'Japan', 'East Asia & Pacific', 'High income', 'JPY', 125800000, 'Tokyo'),
('DEU', 'Germany', 'Europe & Central Asia', 'High income', 'EUR', 83200000, 'Berlin'),
('IND', 'India', 'South Asia', 'Lower middle income', 'INR', 1380000000, 'New Delhi'),
('GBR', 'United Kingdom', 'Europe & Central Asia', 'High income', 'GBP', 67500000, 'London'),
('FRA', 'France', 'Europe & Central Asia', 'High income', 'EUR', 67400000, 'Paris'),
('ITA', 'Italy', 'Europe & Central Asia', 'High income', 'EUR', 59500000, 'Rome'),
('BRA', 'Brazil', 'Latin America & Caribbean', 'Upper middle income', 'BRL', 212600000, 'Brasília'),
('CAN', 'Canada', 'North America', 'High income', 'CAD', 38000000, 'Ottawa'),
('RUS', 'Russian Federation', 'Europe & Central Asia', 'Upper middle income', 'RUB', 146000000, 'Moscow'),
('KOR', 'Korea, Rep.', 'East Asia & Pacific', 'High income', 'KRW', 51300000, 'Seoul'),
('AUS', 'Australia', 'East Asia & Pacific', 'High income', 'AUD', 25700000, 'Canberra'),
('ESP', 'Spain', 'Europe & Central Asia', 'High income', 'EUR', 47400000, 'Madrid'),
('MEX', 'Mexico', 'Latin America & Caribbean', 'Upper middle income', 'MXN', 128900000, 'Mexico City')
ON CONFLICT (code) DO NOTHING;

-- Insert sample data sources
INSERT INTO data_sources (name, source_type, url, api_key_required, update_frequency, status, configuration) VALUES
('World Bank Open Data', 'api', 'https://api.worldbank.org/v2', false, 'daily', 'active', '{"format": "json", "per_page": 1000}'),
('IMF Data', 'api', 'https://dataservices.imf.org/REST/SDMX_JSON.svc', false, 'monthly', 'active', '{"format": "json"}'),
('OECD Data', 'api', 'https://stats.oecd.org/SDMX-JSON', true, 'quarterly', 'active', '{"format": "json"}'),
('Federal Reserve Economic Data', 'api', 'https://api.stlouisfed.org/fred', true, 'daily', 'active', '{"format": "json"}'),
('European Central Bank', 'api', 'https://sdw-wsrest.ecb.europa.eu/service', false, 'daily', 'active', '{"format": "json"}'),
('Reserve Bank of India', 'api', 'https://rbi.org.in/Scripts/api', false, 'weekly', 'active', '{"format": "json"}'),
('Bank of Japan', 'api', 'https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2', false, 'monthly', 'active', '{"format": "json"}')
ON CONFLICT (name) DO NOTHING;

-- Insert sample GDP records
INSERT INTO gdp_records (country_code, period, gdp_value, method, components, quality_score, metadata, created_by) 
SELECT 
    'USA', 
    '2024-Q1', 
    27000.00, 
    'expenditure',
    '{"consumption": 18500, "investment": 4800, "government": 4200, "net_exports": -500}'::jsonb,
    0.95,
    '{"source": "Bureau of Economic Analysis", "methodology": "SNA 2008", "revision": "third_estimate"}'::jsonb,
    (SELECT id FROM users WHERE username = 'admin' LIMIT 1)
WHERE NOT EXISTS (SELECT 1 FROM gdp_records WHERE country_code = 'USA' AND period = '2024-Q1');

-- Create materialized views for better performance
CREATE MATERIALIZED VIEW IF NOT EXISTS gdp_summary AS
SELECT 
    country_code,
    COUNT(*) as total_records,
    MAX(period) as latest_period,
    AVG(gdp_value) as avg_gdp,
    MAX(gdp_value) as max_gdp,
    MIN(gdp_value) as min_gdp,
    AVG(quality_score) as avg_quality_score
FROM gdp_records
GROUP BY country_code;

CREATE UNIQUE INDEX IF NOT EXISTS idx_gdp_summary_country ON gdp_summary(country_code);

-- Create view for latest GDP data
CREATE OR REPLACE VIEW latest_gdp_data AS
SELECT DISTINCT ON (country_code)
    country_code,
    period,
    gdp_value,
    method,
    components,
    quality_score,
    created_at
FROM gdp_records
ORDER BY country_code, created_at DESC;

-- Create functions for data validation
CREATE OR REPLACE FUNCTION validate_gdp_components(components jsonb, method text)
RETURNS boolean AS $$
BEGIN
    CASE method
        WHEN 'expenditure' THEN
            RETURN components ? 'consumption' AND components ? 'investment' 
                   AND components ? 'government' AND components ? 'net_exports';
        WHEN 'income' THEN
            RETURN components ? 'wages_salaries' AND components ? 'corporate_profits';
        WHEN 'output' THEN
            RETURN components ? 'agriculture' AND components ? 'manufacturing' 
                   AND components ? 'services';
        ELSE
            RETURN false;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Create audit table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, user_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), OLD.created_by);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, new_values, user_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW), NEW.created_by);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_values, user_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW), NEW.created_by);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers
CREATE TRIGGER gdp_records_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON gdp_records
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create admin user
INSERT INTO users (username, email, password_hash, first_name, last_name, role, is_active)
VALUES (
    'admin',
    'admin@gdp-platform.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewYhtHfKzLuB1j8m', -- password: admin123
    'System',
    'Administrator',
    'admin',
    true
) ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres;
