-- ============================================================================
-- DPI-600 DRUG PROFILE INTELLIGENCE SYSTEM
-- Database Schema: Organizations & Case Matching
-- Version: 1.0
-- Created: December 2025
-- Author: Autthapol Saiyat (Boy)
-- ============================================================================

-- Enable PostGIS extension for geographic data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For text similarity search

-- ============================================================================
-- SECTION 1: ORGANIZATION / NETWORK PROFILES
-- ============================================================================

-- Drug trafficking organizations/networks
CREATE TABLE organizations (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,           -- e.g., 'ORG-001', 'NET-LION'
    name VARCHAR(200) NOT NULL,                 -- e.g., 'กลุ่มเครือข่ายสิงโตทอง'
    aliases TEXT[],                             -- Alternative names
    description TEXT,
    
    -- Geographic info
    primary_region VARCHAR(100),                -- e.g., 'ภาคเหนือ'
    active_provinces TEXT[],                    -- e.g., ['เชียงราย', 'เชียงใหม่']
    known_routes TEXT[],                        -- Trafficking routes
    center_point GEOGRAPHY(POINT, 4326),        -- Geographic center for mapping
    
    -- Drug characteristics
    primary_drug_types TEXT[],                  -- e.g., ['methamphetamine', 'yaba']
    known_logos TEXT[],                         -- e.g., ['lion', 'tiger', 'wy']
    chemical_signature JSONB,                   -- Typical chemical composition
    
    -- Intelligence
    threat_level VARCHAR(20) DEFAULT 'medium',  -- low, medium, high, critical
    estimated_members INTEGER,
    known_leaders TEXT[],
    international_connections TEXT[],
    
    -- Statistics
    total_cases INTEGER DEFAULT 0,
    total_seizure_weight_kg DECIMAL(12,3) DEFAULT 0,
    total_seizure_pills INTEGER DEFAULT 0,
    first_seen_date DATE,
    last_seen_date DATE,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',        -- active, dormant, disbanded
    confidence_score DECIMAL(5,2),              -- How confident we are in this profile
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER,
    notes TEXT
);

-- Organization activity timeline
CREATE TABLE organization_activities (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    activity_date DATE NOT NULL,
    activity_type VARCHAR(50),                  -- seizure, arrest, sighting, intelligence
    description TEXT,
    location VARCHAR(200),
    location_point GEOGRAPHY(POINT, 4326),
    source VARCHAR(100),                        -- Source of intelligence
    reliability VARCHAR(20),                    -- confirmed, probable, unconfirmed
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chemical signatures for organizations (historical data)
CREATE TABLE organization_chemical_profiles (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    sample_date DATE,
    methamphetamine_pct DECIMAL(5,2),
    caffeine_pct DECIMAL(5,2),
    lactose_pct DECIMAL(5,2),
    mdma_pct DECIMAL(5,2),
    other_compounds JSONB,                      -- Other detected compounds
    sample_source VARCHAR(100),                 -- Case reference
    lab_id VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 2: DRUG CASES
-- ============================================================================

-- Main drug cases table
CREATE TABLE drug_cases (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(50) UNIQUE NOT NULL,    -- e.g., 'DPI-2568-001234'
    
    -- Basic info
    case_date DATE NOT NULL,
    case_time TIME,
    case_type VARCHAR(50),                      -- seizure, arrest, investigation
    status VARCHAR(30) DEFAULT 'open',          -- open, closed, pending
    
    -- Location
    province VARCHAR(100) NOT NULL,
    district VARCHAR(100),
    subdistrict VARCHAR(100),
    address TEXT,
    location_point GEOGRAPHY(POINT, 4326),
    region VARCHAR(50),                         -- ภาคเหนือ, ภาคอีสาน, etc.
    
    -- Drug info
    drug_type VARCHAR(100),                     -- methamphetamine, heroin, etc.
    detected_logo VARCHAR(100),                 -- AI detected logo
    logo_confidence DECIMAL(5,2),               -- AI confidence %
    quantity_pills INTEGER,
    quantity_weight_kg DECIMAL(10,3),
    estimated_value_thb DECIMAL(15,2),
    
    -- Chemical analysis
    chemical_profile JSONB,                     -- Lab results
    lab_analysis_date DATE,
    lab_reference VARCHAR(100),
    
    -- Images
    primary_image_url TEXT,
    image_urls TEXT[],
    
    -- Investigation
    investigating_unit VARCHAR(200),
    lead_officer VARCHAR(200),
    officer_rank VARCHAR(50),
    officer_phone VARCHAR(20),
    
    -- Suspects
    suspects_count INTEGER DEFAULT 0,
    suspects_info JSONB,                        -- Anonymized suspect data
    
    -- F-NET Integration
    fnet_case_id VARCHAR(100),
    fnet_sync_date TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER,
    notes TEXT
);

-- Case images table
CREATE TABLE case_images (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES drug_cases(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    image_type VARCHAR(50),                     -- primary, evidence, scene, suspect
    capture_date TIMESTAMP,
    capture_device VARCHAR(100),
    gps_location GEOGRAPHY(POINT, 4326),
    ai_processed BOOLEAN DEFAULT FALSE,
    ai_detected_logo VARCHAR(100),
    ai_confidence DECIMAL(5,2),
    ai_processed_at TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 3: CASE-ORGANIZATION MATCHING
-- ============================================================================

-- Links cases to potential organizations
CREATE TABLE case_organization_matches (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES drug_cases(id) ON DELETE CASCADE,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Match scores (0-100)
    logo_match_score DECIMAL(5,2),              -- AI logo similarity
    chemical_match_score DECIMAL(5,2),          -- Chemical signature similarity
    geographic_match_score DECIMAL(5,2),        -- Location correlation
    temporal_match_score DECIMAL(5,2),          -- Time pattern correlation
    overall_confidence DECIMAL(5,2),            -- Weighted overall score
    
    -- Match details
    match_algorithm_version VARCHAR(20),
    match_factors JSONB,                        -- Detailed breakdown
    
    -- Verification
    verified_by_analyst BOOLEAN DEFAULT FALSE,
    analyst_id INTEGER,
    analyst_notes TEXT,
    verification_date TIMESTAMP,
    
    -- Ranking
    rank_position INTEGER,                      -- 1, 2, 3 for top matches
    
    -- Metadata
    matched_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Match history for audit trail
CREATE TABLE match_history (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES drug_cases(id) ON DELETE CASCADE,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    action VARCHAR(50),                         -- created, updated, verified, rejected
    previous_score DECIMAL(5,2),
    new_score DECIMAL(5,2),
    changed_by INTEGER,
    change_reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 4: LOGO REFERENCE
-- ============================================================================

-- Known drug logos/stamps
CREATE TABLE drug_logos (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,           -- e.g., 'lion', 'wy', '999'
    name VARCHAR(100) NOT NULL,
    name_th VARCHAR(100),                       -- Thai name
    description TEXT,
    
    -- Visual info
    reference_image_url TEXT,
    color_primary VARCHAR(50),
    color_secondary VARCHAR(50),
    shape VARCHAR(50),                          -- round, oval, rectangle
    
    -- Statistics
    first_seen_date DATE,
    last_seen_date DATE,
    total_cases INTEGER DEFAULT 0,
    primary_regions TEXT[],
    
    -- AI Model info
    model_class_id INTEGER,                     -- Class ID in AI model
    model_accuracy DECIMAL(5,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Logo-Organization associations
CREATE TABLE logo_organization_links (
    id SERIAL PRIMARY KEY,
    logo_id INTEGER REFERENCES drug_logos(id) ON DELETE CASCADE,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
    association_strength VARCHAR(20),           -- primary, secondary, occasional
    first_associated DATE,
    last_associated DATE,
    case_count INTEGER DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 5: GEOGRAPHIC REGIONS
-- ============================================================================

-- Thai provinces with regions
CREATE TABLE provinces (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name_th VARCHAR(100) NOT NULL,
    name_en VARCHAR(100),
    region VARCHAR(50),                         -- ภาคเหนือ, ภาคอีสาน, etc.
    center_point GEOGRAPHY(POINT, 4326),
    boundary GEOGRAPHY(POLYGON, 4326),
    is_border_province BOOLEAN DEFAULT FALSE,
    border_countries TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Forensic centers
CREATE TABLE forensic_centers (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    region VARCHAR(50),
    address TEXT,
    location_point GEOGRAPHY(POINT, 4326),
    responsible_provinces TEXT[],
    phone VARCHAR(50),
    email VARCHAR(100),
    head_officer VARCHAR(200),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 6: USERS & AUDIT
-- ============================================================================

-- System users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(200),
    
    -- Profile
    full_name VARCHAR(200),
    rank VARCHAR(50),
    position VARCHAR(200),
    unit VARCHAR(200),
    phone VARCHAR(50),
    
    -- Access
    role VARCHAR(50) DEFAULT 'viewer',          -- admin, analyst, officer, viewer
    permissions JSONB,
    forensic_center_id INTEGER REFERENCES forensic_centers(id),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Audit log
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id INTEGER,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 7: INDEXES
-- ============================================================================

-- Organizations
CREATE INDEX idx_org_region ON organizations(primary_region);
CREATE INDEX idx_org_status ON organizations(status);
CREATE INDEX idx_org_threat ON organizations(threat_level);
CREATE INDEX idx_org_location ON organizations USING GIST(center_point);

-- Cases
CREATE INDEX idx_case_date ON drug_cases(case_date);
CREATE INDEX idx_case_province ON drug_cases(province);
CREATE INDEX idx_case_region ON drug_cases(region);
CREATE INDEX idx_case_logo ON drug_cases(detected_logo);
CREATE INDEX idx_case_status ON drug_cases(status);
CREATE INDEX idx_case_location ON drug_cases USING GIST(location_point);
CREATE INDEX idx_case_number ON drug_cases(case_number);

-- Matches
CREATE INDEX idx_match_case ON case_organization_matches(case_id);
CREATE INDEX idx_match_org ON case_organization_matches(organization_id);
CREATE INDEX idx_match_confidence ON case_organization_matches(overall_confidence DESC);
CREATE INDEX idx_match_rank ON case_organization_matches(case_id, rank_position);

-- Logos
CREATE INDEX idx_logo_code ON drug_logos(code);

-- Full text search
CREATE INDEX idx_org_name_search ON organizations USING gin(name gin_trgm_ops);
CREATE INDEX idx_case_notes_search ON drug_cases USING gin(notes gin_trgm_ops);

-- ============================================================================
-- SECTION 8: VIEWS
-- ============================================================================

-- Top organization matches for each case
CREATE OR REPLACE VIEW v_case_top_matches AS
SELECT 
    c.id AS case_id,
    c.case_number,
    c.case_date,
    c.province,
    c.detected_logo,
    o.id AS org_id,
    o.name AS org_name,
    o.primary_region,
    m.logo_match_score,
    m.chemical_match_score,
    m.geographic_match_score,
    m.overall_confidence,
    m.rank_position
FROM drug_cases c
JOIN case_organization_matches m ON c.id = m.case_id
JOIN organizations o ON m.organization_id = o.id
WHERE m.rank_position <= 3
ORDER BY c.case_date DESC, m.rank_position;

-- Organization statistics
CREATE OR REPLACE VIEW v_organization_stats AS
SELECT 
    o.id,
    o.code,
    o.name,
    o.primary_region,
    o.threat_level,
    o.status,
    COUNT(DISTINCT m.case_id) AS matched_cases,
    AVG(m.overall_confidence) AS avg_confidence,
    MAX(c.case_date) AS last_activity,
    array_agg(DISTINCT c.province) AS affected_provinces
FROM organizations o
LEFT JOIN case_organization_matches m ON o.id = m.organization_id AND m.rank_position = 1
LEFT JOIN drug_cases c ON m.case_id = c.id
GROUP BY o.id, o.code, o.name, o.primary_region, o.threat_level, o.status;

-- Monthly case summary by region
CREATE OR REPLACE VIEW v_monthly_case_summary AS
SELECT 
    DATE_TRUNC('month', case_date) AS month,
    region,
    COUNT(*) AS total_cases,
    SUM(quantity_pills) AS total_pills,
    SUM(quantity_weight_kg) AS total_weight_kg,
    array_agg(DISTINCT detected_logo) AS logos_found
FROM drug_cases
WHERE case_date >= NOW() - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', case_date), region
ORDER BY month DESC, region;

-- ============================================================================
-- SECTION 9: FUNCTIONS
-- ============================================================================

-- Calculate chemical similarity between two profiles
CREATE OR REPLACE FUNCTION calculate_chemical_similarity(
    profile1 JSONB,
    profile2 JSONB
) RETURNS DECIMAL(5,2) AS $$
DECLARE
    meth_diff DECIMAL;
    caff_diff DECIMAL;
    lact_diff DECIMAL;
    mdma_diff DECIMAL;
    total_diff DECIMAL;
    similarity DECIMAL;
BEGIN
    meth_diff := ABS(COALESCE((profile1->>'meth')::DECIMAL, 0) - COALESCE((profile2->>'meth')::DECIMAL, 0));
    caff_diff := ABS(COALESCE((profile1->>'caff')::DECIMAL, 0) - COALESCE((profile2->>'caff')::DECIMAL, 0));
    lact_diff := ABS(COALESCE((profile1->>'lact')::DECIMAL, 0) - COALESCE((profile2->>'lact')::DECIMAL, 0));
    mdma_diff := ABS(COALESCE((profile1->>'mdma')::DECIMAL, 0) - COALESCE((profile2->>'mdma')::DECIMAL, 0));
    
    total_diff := SQRT(meth_diff^2 + caff_diff^2 + lact_diff^2 + mdma_diff^2);
    similarity := GREATEST(0, 100 - total_diff);
    
    RETURN ROUND(similarity, 2);
END;
$$ LANGUAGE plpgsql;

-- Calculate overall match score
CREATE OR REPLACE FUNCTION calculate_match_score(
    logo_score DECIMAL,
    chemical_score DECIMAL,
    geo_score DECIMAL,
    temporal_score DECIMAL DEFAULT NULL
) RETURNS DECIMAL(5,2) AS $$
BEGIN
    IF temporal_score IS NOT NULL THEN
        RETURN ROUND((logo_score * 0.35 + chemical_score * 0.35 + geo_score * 0.15 + temporal_score * 0.15), 2);
    ELSE
        RETURN ROUND((logo_score * 0.40 + chemical_score * 0.40 + geo_score * 0.20), 2);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER tr_org_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_case_updated_at BEFORE UPDATE ON drug_cases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_match_updated_at BEFORE UPDATE ON case_organization_matches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- SECTION 10: SAMPLE DATA
-- ============================================================================

-- Insert sample organizations
INSERT INTO organizations (code, name, aliases, primary_region, active_provinces, known_logos, chemical_signature, threat_level, total_cases, first_seen_date, last_seen_date, status) VALUES
('ORG-001', 'กลุ่มเครือข่าย "สิงโตทอง"', ARRAY['Golden Lion', 'เครือข่ายเหนือ'], 'ภาคเหนือ', ARRAY['เชียงราย', 'เชียงใหม่', 'แม่ฮ่องสอน'], ARRAY['lion', 'tiger'], '{"meth": 85, "caff": 40, "lact": 20, "mdma": 5}', 'high', 47, '2566-01-15', '2568-12-01', 'active'),
('ORG-002', 'กลุ่มเครือข่าย "WY International"', ARRAY['WY Group', 'เครือข่ายอีสาน'], 'ภาคอีสาน', ARRAY['นครพนม', 'มุกดาหาร', 'อุบลราชธานี'], ARRAY['wy', 'yaba'], '{"meth": 78, "caff": 55, "lact": 25, "mdma": 8}', 'high', 23, '2567-03-20', '2568-11-28', 'active'),
('ORG-003', 'กลุ่มเครือข่าย "ชายแดนใต้"', ARRAY['Southern Border', 'เครือข่ายใต้'], 'ภาคกลาง', ARRAY['กรุงเทพมหานคร', 'สมุทรปราการ', 'ชลบุรี'], ARRAY['999', 'star'], '{"meth": 70, "caff": 30, "lact": 35, "mdma": 15}', 'medium', 12, '2568-02-10', '2568-11-15', 'active'),
('ORG-004', 'กลุ่มเครือข่าย "ม้าเหล็ก"', ARRAY['Iron Horse', 'เครือข่ายตะวันตก'], 'ภาคตะวันตก', ARRAY['กาญจนบุรี', 'ตาก', 'ราชบุรี'], ARRAY['horse', 'eagle'], '{"meth": 82, "caff": 35, "lact": 28, "mdma": 3}', 'medium', 18, '2566-08-05', '2568-10-20', 'active'),
('ORG-005', 'กลุ่มเครือข่าย "R-Network"', ARRAY['R Group', 'เครือข่าย R'], 'ภาคใต้', ARRAY['สงขลา', 'นราธิวาส', 'ยะลา'], ARRAY['r', 'star'], '{"meth": 75, "caff": 45, "lact": 22, "mdma": 12}', 'medium', 15, '2567-01-10', '2568-09-30', 'active');

-- Insert sample logos
INSERT INTO drug_logos (code, name, name_th, shape, primary_regions, total_cases, model_class_id) VALUES
('lion', 'Lion', 'สิงโต', 'round', ARRAY['ภาคเหนือ'], 47, 0),
('wy', 'WY', 'ดับเบิลยูวาย', 'round', ARRAY['ภาคอีสาน'], 23, 1),
('999', 'Triple Nine', 'สามเก้า', 'round', ARRAY['ภาคกลาง'], 12, 2),
('horse', 'Horse', 'ม้า', 'oval', ARRAY['ภาคตะวันตก'], 18, 3),
('r', 'R', 'อาร์', 'round', ARRAY['ภาคใต้'], 15, 4),
('star', 'Star', 'ดาว', 'round', ARRAY['ภาคกลาง', 'ภาคใต้'], 8, 5),
('tiger', 'Tiger', 'เสือ', 'round', ARRAY['ภาคเหนือ'], 10, 6),
('eagle', 'Eagle', 'นกอินทรี', 'round', ARRAY['ภาคตะวันตก'], 7, 7),
('yaba', 'Yaba', 'ยาบ้า', 'round', ARRAY['ภาคอีสาน'], 5, 8),
('nologo', 'No Logo', 'ไม่มีโลโก้', 'round', ARRAY['ทั่วประเทศ'], 20, 9);

-- Insert sample provinces
INSERT INTO provinces (code, name_th, name_en, region, is_border_province, border_countries) VALUES
('50', 'เชียงใหม่', 'Chiang Mai', 'ภาคเหนือ', TRUE, ARRAY['Myanmar']),
('57', 'เชียงราย', 'Chiang Rai', 'ภาคเหนือ', TRUE, ARRAY['Myanmar', 'Laos']),
('58', 'แม่ฮ่องสอน', 'Mae Hong Son', 'ภาคเหนือ', TRUE, ARRAY['Myanmar']),
('48', 'นครพนม', 'Nakhon Phanom', 'ภาคอีสาน', TRUE, ARRAY['Laos']),
('49', 'มุกดาหาร', 'Mukdahan', 'ภาคอีสาน', TRUE, ARRAY['Laos']),
('34', 'อุบลราชธานี', 'Ubon Ratchathani', 'ภาคอีสาน', TRUE, ARRAY['Laos']),
('10', 'กรุงเทพมหานคร', 'Bangkok', 'ภาคกลาง', FALSE, NULL),
('11', 'สมุทรปราการ', 'Samut Prakan', 'ภาคกลาง', FALSE, NULL),
('20', 'ชลบุรี', 'Chonburi', 'ภาคตะวันออก', FALSE, NULL),
('90', 'สงขลา', 'Songkhla', 'ภาคใต้', TRUE, ARRAY['Malaysia']),
('96', 'นราธิวาส', 'Narathiwat', 'ภาคใต้', TRUE, ARRAY['Malaysia']);

-- Insert sample forensic centers
INSERT INTO forensic_centers (code, name, region, responsible_provinces, status) VALUES
('FC-01', 'ศูนย์พิสูจน์หลักฐาน 1', 'ภาคกลาง', ARRAY['กรุงเทพมหานคร', 'นนทบุรี', 'ปทุมธานี'], 'active'),
('FC-02', 'ศูนย์พิสูจน์หลักฐาน 2', 'ภาคกลาง', ARRAY['สมุทรปราการ', 'สมุทรสาคร', 'สมุทรสงคราม'], 'active'),
('FC-03', 'ศูนย์พิสูจน์หลักฐาน 3', 'ภาคเหนือ', ARRAY['เชียงใหม่', 'เชียงราย', 'ลำพูน'], 'active'),
('FC-04', 'ศูนย์พิสูจน์หลักฐาน 4', 'ภาคเหนือ', ARRAY['แม่ฮ่องสอน', 'ลำปาง', 'แพร่'], 'active'),
('FC-05', 'ศูนย์พิสูจน์หลักฐาน 5', 'ภาคอีสาน', ARRAY['นครราชสีมา', 'บุรีรัมย์', 'สุรินทร์'], 'active'),
('FC-06', 'ศูนย์พิสูจน์หลักฐาน 6', 'ภาคอีสาน', ARRAY['อุบลราชธานี', 'นครพนม', 'มุกดาหาร'], 'active'),
('FC-07', 'ศูนย์พิสูจน์หลักฐาน 7', 'ภาคตะวันออก', ARRAY['ชลบุรี', 'ระยอง', 'จันทบุรี'], 'active'),
('FC-08', 'ศูนย์พิสูจน์หลักฐาน 8', 'ภาคตะวันตก', ARRAY['กาญจนบุรี', 'ราชบุรี', 'เพชรบุรี'], 'active'),
('FC-09', 'ศูนย์พิสูจน์หลักฐาน 9', 'ภาคใต้', ARRAY['สงขลา', 'พัทลุง', 'สตูล'], 'active'),
('FC-10', 'ศูนย์พิสูจน์หลักฐาน 10', 'ภาคใต้', ARRAY['นราธิวาส', 'ยะลา', 'ปัตตานี'], 'active'),
('FC-11', 'ศูนย์พิสูจน์หลักฐาน 11', 'ภาคใต้', ARRAY['ภูเก็ต', 'กระบี่', 'พังงา'], 'active');

-- Insert sample drug cases
INSERT INTO drug_cases (case_number, case_date, province, region, drug_type, detected_logo, logo_confidence, quantity_pills, chemical_profile, status) VALUES
('DPI-2568-000001', '2568-12-01', 'เชียงราย', 'ภาคเหนือ', 'methamphetamine', 'lion', 97.5, 50000, '{"meth": 86, "caff": 38, "lact": 21, "mdma": 4}', 'closed'),
('DPI-2568-000002', '2568-12-03', 'นครพนม', 'ภาคอีสาน', 'methamphetamine', 'wy', 92.3, 30000, '{"meth": 79, "caff": 54, "lact": 24, "mdma": 7}', 'closed'),
('DPI-2568-000003', '2568-12-05', 'กรุงเทพมหานคร', 'ภาคกลาง', 'methamphetamine', '999', 88.7, 15000, '{"meth": 71, "caff": 32, "lact": 33, "mdma": 14}', 'open'),
('DPI-2568-000004', '2568-12-07', 'เชียงใหม่', 'ภาคเหนือ', 'methamphetamine', 'lion', 95.2, 80000, '{"meth": 84, "caff": 41, "lact": 19, "mdma": 6}', 'open'),
('DPI-2568-000005', '2568-12-09', 'มุกดาหาร', 'ภาคอีสาน', 'methamphetamine', 'wy', 89.8, 25000, '{"meth": 77, "caff": 56, "lact": 26, "mdma": 9}', 'open');

-- Insert sample matches
INSERT INTO case_organization_matches (case_id, organization_id, logo_match_score, chemical_match_score, geographic_match_score, overall_confidence, rank_position) VALUES
(1, 1, 97.5, 94.2, 95.0, 95.5, 1),
(1, 2, 45.2, 65.3, 30.0, 48.8, 2),
(1, 4, 38.1, 58.7, 25.0, 42.3, 3),
(2, 2, 92.3, 91.5, 95.0, 92.4, 1),
(2, 1, 42.1, 68.4, 28.0, 48.9, 2),
(2, 3, 35.8, 55.2, 32.0, 42.6, 3),
(3, 3, 88.7, 89.3, 90.0, 89.2, 1),
(3, 2, 41.5, 62.8, 35.0, 47.5, 2),
(3, 5, 38.2, 58.1, 40.0, 45.2, 3),
(4, 1, 95.2, 93.8, 95.0, 94.4, 1),
(4, 4, 42.3, 67.5, 30.0, 48.1, 2),
(4, 2, 38.7, 62.1, 25.0, 43.3, 3),
(5, 2, 89.8, 90.2, 95.0, 90.8, 1),
(5, 1, 43.5, 66.8, 28.0, 48.2, 2),
(5, 3, 36.2, 58.9, 32.0, 43.7, 3);

-- ============================================================================
-- SECTION 11: GRANTS (Example)
-- ============================================================================

-- Create roles
-- CREATE ROLE dpi_admin;
-- CREATE ROLE dpi_analyst;
-- CREATE ROLE dpi_officer;
-- CREATE ROLE dpi_viewer;

-- Grant permissions (example)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO dpi_admin;
-- GRANT SELECT, INSERT, UPDATE ON drug_cases, case_images TO dpi_analyst;
-- GRANT SELECT, INSERT ON drug_cases TO dpi_officer;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO dpi_viewer;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
