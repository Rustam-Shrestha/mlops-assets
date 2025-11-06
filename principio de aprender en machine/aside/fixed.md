# server.js


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
    contactEmail TEXT
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
app.post('/submit-form', (req, res) => {
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
      contactEmail TEXT
    )
  `);

  // Insert into main + facility database
  getNextTicketNumber((err, ticketNumber) => {
    if (err) return res.status(500).json({ success: false, message: err.message });

    const values = [
      ticketNumber, facilityId, data.date, data.time, data.address,
      data.issuesOn, data.issueDescription, data.priority,
      data.contactName, data.contactPhone, data.contactEmail
    ];

    const insertSQL = `
       INSERT INTO requests (
    ticketNumber, facilityId, date, time, address,
    issuesOn, issueDescription, priority,
    contactName, contactPhone, contactEmail
  )
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    db.run(insertSQL, values, (err) => {
      if (err) return res.status(500).json({ success: false, message: err.message });

      // Insert same record into facility DB
      facilityDB.run(`
         INSERT INTO requests (
    ticketNumber, facilityId, date, time, address,
    issuesOn, issueDescription, priority,
    contactName, contactPhone, contactEmail
  )
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `, [
        ticketNumber, data.date, data.time, data.address,
        data.issuesOn, data.issueDescription, data.priority,
        data.contactName, data.contactPhone, data.contactEmail
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


                       



# index.html


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Request Form</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <h1>BR Compliance and Service</h1>
        <h2>Your Trust is Our Goal</h2>
        <h3>Service Request Form</h3>

        <!-- Form -->
        <form id="serviceForm">
            <!-- Facility ID -->
            <label for="facilityId">Facility ID</label>
            <input type="number" id="facilityId" name="facilityId" placeholder="Enter Facility ID" required>

            <!-- Date -->
            <label for="date">Date</label>
            <input type="date" id="date" name="date" required>

            <!-- Time -->
            <label for="time">Time</label>
            <input type="time" id="time" name="time" required>

            <!-- Address -->
            <label for="address">Address</label>
            <input type="text" id="address" name="address" placeholder="Enter Address" required>

            <!-- Issues On -->
            <label for="issuesOn">Issues On</label>
            <input type="text" id="issuesOn" name="issuesOn" placeholder="Enter Issue Title" required>

            <!-- Issue Description -->
            <label for="issueDescription">Issue Description</label>
            <textarea id="issueDescription" name="issueDescription" rows="5" placeholder="Describe the issue..." required></textarea>

            <!-- Priority Level -->
            <label for="priority">Priority Level</label>
            <select id="priority" name="priority" required>
                <option value="" disabled selected>Select Priority</option>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
                <option value="Urgent">Urgent (After-hours charge will be applied)</option>
            </select>

            <!-- Contact Info -->
            <label for="contactName">Contact Name</label>
            <input type="text" id="contactName" name="contactName" placeholder="Enter Your Name" required>

            <label for="contactPhone">Contact Phone</label>
            <input type="tel" id="contactPhone" name="contactPhone" placeholder="Enter Phone Number" required>

            <label for="contactEmail">Contact Email</label>
            <input type="email" id="contactEmail" name="contactEmail" placeholder="Enter Email" required>

            <button type="submit">Submit Request</button>
        </form>

        <!-- Success/Error Message -->
        <div id="responseMessage"></div>
    </div>

    <script>

document.getElementById('date').valueAsDate = new Date();
const now = new Date();
document.getElementById('time').value = now.toTimeString().slice(0,5);

const form = document.getElementById('serviceForm');
const responseMessage = document.getElementById('responseMessage');

form.addEventListener('submit', async function(e) {
  e.preventDefault();

  const formData = Object.fromEntries(new FormData(form).entries());

  try {
    const res = await fetch('/submit-form', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });

    const result = await res.json();
    if (result.success) {
      responseMessage.textContent = `✅ Request saved! Ticket ID: ${result.ticketNumber}`;
      responseMessage.style.color = 'green';
      form.reset();
    } else {
      responseMessage.textContent = '❌ Failed to save request.';
      responseMessage.style.color = 'red';
    }
  } catch (error) {
    responseMessage.textContent = '⚠️ Error connecting to server.';
    responseMessage.style.color = 'orange';
  }
});
</script>

    </script>
</body>
</html>

