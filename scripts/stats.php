<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// $db = new SQLite3('./scripts/birds.db', SQLITE3_OPEN_CREATE | SQLITE3_OPEN_READWRITE);
$db = new SQLite3('./scripts/detections_whale.db', SQLITE3_OPEN_CREATE | SQLITE3_OPEN_READWRITE);
if($db == False) {
	echo "Database busy";
	header("refresh: 0;");
}
// $statement = $db->prepare('SELECT Date, Time, File_Name, Com_Name, COUNT(*), MAX(Confidence) FROM detections GROUP BY Com_Name ORDER BY COUNT(*) DESC');
// if($statement == False) {
// 	echo "Database busy";
// 	header("refresh: 0;");
// }
// $statement = $db->prepare(
//   "SELECT Date, Time, File_Name, Com_Name,
//           COUNT(*) AS Occurrences,
//           ROUND(100.0 * MAX(CAST(Score AS REAL))) AS MaxConfidence
//    FROM detections
//    GROUP BY Com_Name
//    ORDER BY Occurrences DESC"
// );
$statement = $db->prepare("SELECT Date, Time, File_Name, Com_Name, COUNT(*) AS Occurrences, ROUND(CASE WHEN Confidence IS NULL THEN NULL WHEN Confidence <= 1.0 THEN 100.0 * MAX(CAST(Confidence AS REAL)) ELSE MAX(CAST(Confidence AS REAL)) END) AS MaxConfidence FROM detections GROUP BY Com_Name ORDER BY Occurrences DESC");

$result = $statement->execute();

// $statement2 = $db->prepare('SELECT Date, Time, File_Name, Com_Name, COUNT(*), MAX(Confidence) FROM detections GROUP BY Com_Name ORDER BY Com_Name');
// if($statement == False) {
// 	echo "Database busy";
// 	header("refresh: 0;");
// }
// $statement2 = $db->prepare(
//   "SELECT Date, Time, File_Name, Com_Name,
//           COUNT(*) AS Occurrences,
//           ROUND(100.0 * MAX(CAST(Score AS REAL))) AS MaxConfidence
//    FROM detections
//    GROUP BY Com_Name
//    ORDER BY Com_Name"
// );
$statement2 = $db->prepare("SELECT Date, Time, File_Name, Com_Name, COUNT(*) AS Occurrences, ROUND(CASE WHEN Confidence IS NULL THEN NULL WHEN Confidence <= 1.0 THEN 100.0 * MAX(CAST(Confidence AS REAL)) ELSE MAX(CAST(Confidence AS REAL)) END) AS MaxConfidence FROM detections GROUP BY Com_Name ORDER BY Com_Name");

$result2 = $statement2->execute();



if(isset($_POST['species'])){
  $selection = $_POST['species'];
  // $statement3 = $db->prepare("
  //   SELECT d.Com_Name,
  //          d.Sci_Name,
  //          c.Occurrences,
  //          ROUND(100.0 * c.MaxScore) AS MaxConfidence,
  //          d.File_Name, d.Date, d.Time
  //   FROM detections AS d
  //   JOIN (
  //     SELECT Com_Name,
  //            COUNT(*) AS Occurrences,
  //            MAX(CAST(Score AS REAL)) AS MaxScore
  //     FROM detections
  //     WHERE Com_Name = :species
  //   ) AS c
  //     ON c.Com_Name = d.Com_Name
  //   WHERE d.Com_Name = :species
  //     AND CAST(d.Score AS REAL) = c.MaxScore
  //   ORDER BY d.Date DESC, d.Time DESC
  //   LIMIT 1
  // ");
  $statement3 = $db->prepare("SELECT d.Com_Name, d.Sci_Name, c.Occurrences, ROUND(CASE WHEN c.MaxConfidence IS NULL THEN NULL WHEN c.MaxConfidence <= 1.0 THEN 100.0 * c.MaxConfidence ELSE c.MaxConfidence END) AS MaxConfidence, d.File_Name, d.Date, d.Time FROM detections AS d JOIN (SELECT Com_Name, COUNT(*) AS Occurrences, MAX(CAST(Confidence AS REAL)) AS MaxConfidence FROM detections WHERE Com_Name = :species) AS c ON c.Com_Name = d.Com_Name WHERE d.Com_Name = :species AND CAST(d.Confidence AS REAL) = c.MaxConfidence ORDER BY d.Date DESC, d.Time DESC LIMIT 1");

  if (!$statement3) { die("Database busy"); }
  $statement3->bindValue(':species', $selection, SQLITE3_TEXT);
  $result3 = $statement3->execute();
  // $statement3 = $db->prepare("SELECT Com_Name, Sci_Name, COUNT(*), MAX(Confidence), File_Name, Date, Time from detections WHERE Com_Name = \"$selection\"");
  // if($statement3 == False) {
  // 	echo "Database busy";
  // 	header("refresh: 0;");
  // }
  // $result3 = $statement3->execute();
}
?>

<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BirdNET-Pi DB</title>
<style>
</style>

</head>
<body>
<div class="stats">
<div class="column left">
<table>
<?php
while($results=$result2->fetchArray(SQLITE3_ASSOC))
{
$comname = preg_replace('/ /', '_', $results['Com_Name']);
$comname = preg_replace('/\'/', '', $comname);
$filename = "/By_Date/".$results['Date']."/".$comname."/".$results['File_Name'];
?>
  <tr>
  <form action="" method="POST">
  <td><input type="hidden" name="view" value="Species Stats">
    <button type="submit" name="species" value="<?php echo $results['Com_Name'];?>"><?php echo $results['Com_Name'];?></button>
  </td>
<?php
}
?>
  </form>
  </tr>
</table>
</div>
<div class="column center">
<?php if(!isset($_POST['species'])){
?>
<?php
};?>
<?php if(isset($_POST['species'])){
  $species = $_POST['species'];
   
  while($results = $result3->fetchArray(SQLITE3_ASSOC)) {
    $count = $results['Occurrences'];
    $maxconf = $results['MaxConfidence'];
    $date = $results['Date'];
    $time = $results['Time'];
    $name = $results['Com_Name'];
    $sciname = $results['Sci_Name'];
    $dbsciname = preg_replace('/ /', '_', $sciname);
    $comname = preg_replace('/ /', '_', $results['Com_Name']);
    $comname = preg_replace('/\'/', '', $comname);
    $filename = "/By_Date/".$date."/".$comname."/".$results['File_Name'];
  
    echo "<h3>$name</h3>
      <table><tr>
        <td>
          <i>$sciname</i><br>
          <b>Occurrences:</b> $count<br>
          <b>Max Confidence:</b> $maxconf<br>
          <b>Best Recording:</b> $date $time<br>
          <video controls poster=\"{$filename}.png\" title=\"$filename\">
            <source src=\"$filename\">
          </video>
        </td>
      </tr></table>";
  
    ob_flush();
    flush();
  }
}
  ?>
  <br><br><br>
  <table>
  <?php

while($results=$result->fetchArray(SQLITE3_ASSOC))
{
$comname = preg_replace('/ /', '_', $results['Com_Name']);
$comname = preg_replace('/\'/', '', $comname);
$filename = "/By_Date/".$results['Date']."/".$comname."/".$results['File_Name'];
?>
      <tr>
      <form action="" method="POST">
      <td><input type="hidden" name="view" value="Species Stats">
        <!-- <button type="submit" name="species" value="<?php echo $results['Com_Name'];?>"><?php echo $results['Com_Name'];?></button><br><b>Occurrences:</b> <?php echo $results['COUNT(*)'];?><br>
      <b>Max Confidence:</b> <?php echo $results['MAX(Confidence)'];?><br> -->
        <button type="submit" name="species" value="<?php echo $results['Com_Name'];?>"><?php echo $results['Com_Name'];?></button><br><b>Occurrences:</b> <?php echo $results['Occurrences'];?><br>
      <b>Max Confidence:</b> <?php echo $results['MaxConfidence'];?><br>
      <b>Best Recording:</b> <?php echo $results['Date']." ".$results['Time'];?><br><video controls poster="<?php echo $filename.".png";?>" preload="none" title="<?php echo $filename;?>"><source src="<?php echo $filename;?>" type="audio/mp3"></video></td>
      </tr>
<?php
}
?>
    </table>
      </form>
</div>
</div>
</body>
</html>
