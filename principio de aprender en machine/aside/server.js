const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
const PORT = 3000;
app.use(express.static(path.join(__dirname, 'public')));

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Connect to main DB (on USB)
const mainDBPath = 'C:/media/br1/BRCAS/service_requests.db';
const db = new sqlite3.Database(mainDBPath, (err) => {
  if (err) console.error('❌ SQLite Connection Failed:', err);
  else console.log('✅ Connected to SQLite at', mainDBPath);
});

// Ensure base table exists
db.run(`
  CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticketNumber TEXT,
    facilityId TEXT,
    date TEXT,
    time TEXT,
    address TEXT,
    issuesOn TEXT,
    issueDescription TEXT,
    priority TEXT,
    contactName TEXT,
    contactPhone TEXT,
    contactEmail TEXT,
    storeName TEXT
  )
`);

// Generate next ticket number
function getNextTicketNumber(callback) {
  db.get('SELECT ticketNumber FROM requests ORDER BY id DESC LIMIT 1', [], (err, row) => {
    if (err) return callback(err);
    if (!row) return callback(null, 'BR1001');
    const lastTicket = row.ticketNumber;
    const num = parseInt(lastTicket.replace(/\D/g, '')) + 1;
    callback(null, `BR${num}`);
  });
}

// Handle form submission
app.post('/req-submit', (req, res) => {
  const data = req.body;
  const facilityId = data.facilityId;

  // Create per-facility DB path
const facilityDBPath = `C:/media/br1/BRCAS/facilities/facility_${facilityId}.db`;
  const facilityDB = new sqlite3.Database(facilityDBPath);

  // Create table in facility DB if not exists
  facilityDB.run(`
    CREATE TABLE IF NOT EXISTS service_requests (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ticketNumber TEXT,
      date TEXT,
      time TEXT,
      address TEXT,
      issuesOn TEXT,
      issueDescription TEXT,
      priority TEXT,
      contactName TEXT,
      contactPhone TEXT,
      contactEmail TEXT, 
      storeName TEXT
    )
  `);

  // Insert into main + facility database
  getNextTicketNumber((err, ticketNumber) => {
    if (err) return res.status(500).json({ success: false, message: err.message });

    const values = [
      ticketNumber, facilityId, data.date, data.time, data.address,
      data.issuesOn, data.issueDescription, data.priority,
      data.contactName, data.contactPhone, data.contactEmail, data.storeName
    ];

    const insertSQL = `
       INSERT INTO requests (
    ticketNumber, facilityId, date, time, address,
    issuesOn, issueDescription, priority,
    contactName, contactPhone, contactEmail, storeName
  )
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    db.run(insertSQL, values, (err) => {
      if (err) return res.status(500).json({ success: false, message: err.message });

      // Insert same record into facility DB
      facilityDB.run(`
         INSERT INTO requests (
    ticketNumber, facilityId, date, time, address,
    issuesOn, issueDescription, priority,
    contactName, contactPhone, contactEmail, storeName
  )
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `, [
        ticketNumber, data.date, data.time, data.address,
        data.issuesOn, data.issueDescription, data.priority,
        data.contactName, data.contactPhone, data.contactEmail, data.storeName
      ], (facilityErr) => {
        if (facilityErr) console.error('Facility DB insert error:', facilityErr.message);
      });

      res.json({ success: true, ticketNumber });
    });
  });
});
// Return last 20 requests
app.get('/api/requests', (req, res) => {
  db.all('SELECT * FROM requests ORDER BY id DESC LIMIT 20', [], (err, rows) => {
    if (err) return res.status(500).json({ success: false, message: err.message });
    res.json({ success: true, data: rows });
  });
});

// Serve dashboard
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});


                       
